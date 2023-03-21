import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import List, Optional, Union
import numpy as np
from Attention import Attention


# https://github.com/huggingface/transformers/blob/main/src/transformers/tf_utils.py
def shape_list(tensor: Union[tf.Tensor, np.ndarray]) -> List[int]:
    if isinstance(tensor, np.ndarray):
        return list(tensor.shape)
    dynamic = tf.shape(tensor)
    if tensor.shape == (None):
        return dynamic
    static = tensor.shape.as_list()
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]


class DropPath(tf.keras.layers.Layer):
    def __init__(self, drop_path, **kwargs):
        super().__init__(**kwargs)
        self.drop_path = drop_path

    def call(self, x, training=None):
        if training:
            keep_prob = 1 - self.drop_path
            shape = (tf.shape(x)[0],) + (1,) * (len(tf.shape(x)) - 1)
            random_tensor = keep_prob + tf.random.uniform(shape, 0, 1)
            random_tensor = tf.floor(random_tensor)
            return (x / keep_prob) * random_tensor
        return x


class DWConv(tf.keras.layers.Layer):
    def __init__(self, dim: int = 768, **kwargs):
        super().__init__(**kwargs)
        self.depthwise_convolution = tf.keras.layers.Conv2D(
            filters=dim,
            kernel_size=3,
            strides=1,
            padding="same",
            groups=dim,
        )

    def call(self, x: tf.Tensor, height: int, width: int) -> tf.Tensor:
        batch_size = shape_list(x)[0]
        num_channels = shape_list(x)[-1]
        x = tf.reshape(x, (batch_size, height, width, num_channels))
        x = self.depthwise_convolution(x)

        new_height = shape_list(x)[1]
        new_width = shape_list(x)[2]
        num_channels = shape_list(x)[3]
        x = tf.reshape(x, (batch_size, new_height * new_width, num_channels))
        return x


class mlp(tf.keras.layers.Layer):
    def __init__(self, dim, dropout):
        super().__init__()
        self.fc1 = tf.keras.layers.Dense(
            dim, kernel_initializer=tf.keras.initializers.GlorotNormal()
        )
        self.dwconv = DWConv(dim)
        self.fc2 = tf.keras.layers.Dense(
            dim, kernel_initializer=tf.keras.initializers.GlorotNormal()
        )
        self.drop = tf.keras.layers.Dropout(dropout)

    def call(self, input_tensor, H, W):
        x = self.fc1(input_tensor)
        x = self.dwconv(x, H, W)
        x = tf.nn.gelu(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(tf.keras.layers.Layer):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        sr_ratio=1,
    ):
        super().__init__()
        self.norm1 = tf.keras.layers.LayerNormalization()
        self.Attention = Attention(
            dim, num_heads, sr_ratio, attn_drop
        )
        self.drop_path = DropPath(drop_path)
        self.mlp = mlp(dim * mlp_ratio, drop)
        self.norm2 = tf.keras.layers.LayerNormalization()

    def call(self, x, H, W):
        x = x + self.drop_path(self.Attention(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x

class OverlapPatchEmbed(tf.keras.Model):
    def __init__(self, img_size=224, patch_size=7, stride=4, embed_dim=768, **kwargs):
        super().__init__(**kwargs)
        self.padding = tf.keras.layers.ZeroPadding2D(padding=patch_size // 2)
        self.conv = tf.keras.layers.Conv2D(
            filters=embed_dim, kernel_size=patch_size, strides=stride, padding="VALID")
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-05)

    def call(self, x):
        x = self.conv(self.padding(x))
        H = shape_list(x)[1]
        W = shape_list(x)[2]
        N = shape_list(x)[3]

        x = tf.reshape(x, (-1, H * W, N))
        x = self.norm(x)
        return x, H, W

class MixVisionTransformer(tf.keras.layers.Layer):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dims=[64, 128, 256, 512],
        num_heads=[1, 2, 4, 8],
        mlp_ratios=[4, 4, 4, 4],
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        depths=[3, 4, 6, 3],
        sr_ratios=[8, 4, 2, 1],
    ):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        #patch_embed
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4,
                                              embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2,
                                              embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2,
                                              embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2,
                                              embed_dim=embed_dims[3])

        dpr = [x.numpy() for x in tf.linspace(0.0, drop_path_rate, sum(depths))]
        cur = 0
        self.block1 = []
        for i in range(depths[0]):
            new_block1 = Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i],
            sr_ratio=sr_ratios[0])
            self.block1.append(new_block1)
        self.norm1 = tf.keras.layers.LayerNormalization()

        cur += depths[0]
        self.block2 = []
        for i in range(depths[1]):
            new_block2 = Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i],
            sr_ratio=sr_ratios[1])
            self.block2.append(new_block2)
        self.norm2 = tf.keras.layers.LayerNormalization()

        cur += depths[1]
        self.block3 = []
        for i in range(depths[2]):
            new_block3 = Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i],
            sr_ratio=sr_ratios[2])
            self.block3.append(new_block3)
        self.norm3 = tf.keras.layers.LayerNormalization()

        cur += depths[2]
        self.block4 = []
        for i in range(depths[3]):
            new_block4 = Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i],
            sr_ratio=sr_ratios[3])
            self.block4.append(new_block4)
        self.norm4 = tf.keras.layers.LayerNormalization()

    def call_feature(self, x):
        batch = x.shape[0]
        outs = []
        x, H, W = self.patch_embed1(x)
        for i, blk in enumerate(self.block1):
            x = blk(x, H, W)
        x = self.norm1(x)
        height = width = int(x.shape[1] ** 0.5)
        x = tf.reshape(x, (batch, height, width, -1))
        outs.append(x)

        x, H, W = self.patch_embed2(x)
        for i, blk in enumerate(self.block2):
            x = blk(x, H, W)
        x = self.norm2(x)
        height = width = int(x.shape[1] ** 0.5)
        x = tf.reshape(x, (batch, height, width, -1))
        outs.append(x)

        x, H, W = self.patch_embed3(x)
        for i, blk in enumerate(self.block3):
            x = blk(x, H, W)
        x = self.norm3(x)
        height = width = int(x.shape[1] ** 0.5)
        x = tf.reshape(x, (batch, height, width, -1))
        outs.append(x)

        x, H, W = self.patch_embed4(x)
        for i, blk in enumerate(self.block4):
            x = blk(x, H, W)
        x = self.norm4(x)
        height = width = int(x.shape[1] ** 0.5)
        x = tf.reshape(x, (batch, height, width, -1))
        outs.append(x)

        return outs

    def call(self, x):
        x = self.call_feature(x)
        return x


input_layer = tf.keras.layers.Input(shape=(224, 224, 3))
x = MixVisionTransformer(img_size = 224, patch_size=4, embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)(input_layer)
model = tf.keras.Model(inputs=input_layer, outputs=x)