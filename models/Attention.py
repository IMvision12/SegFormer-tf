import tensorflow as tf
from tensorflow.keras import layers


class Attention(tf.keras.layers.Layer):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        sr_ratio=1,
    ):
        super(Attention, self).__init__()
        assert (
            dim % num_heads == 0
        ), f"dim {dim} should be divisible by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q = layers.Dense(dim, use_bias=qkv_bias)
        self.k = layers.Dense(dim, use_bias=qkv_bias)
        self.v = layers.Dense(dim, use_bias=qkv_bias)
        self.attn_drop = layers.Dropout(attn_drop)
        self.proj = layers.Dense(dim)
        self.proj_drop = layers.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = layers.Conv2D(dim, kernel_size=sr_ratio, strides=sr_ratio)
            self.norm = layers.LayerNormalization()

    def call(self, x, H, W):
        get_shape = tf.shape(x)
        B = get_shape[0]
        N = get_shape[1]
        C = get_shape[2]

        q = self.q(x)
        q = tf.reshape(q, [B, N, self.num_heads, C // self.num_heads])
        q = tf.transpose(q, perm=[0, 2, 1, 3])

        if self.sr_ratio > 1:
            x = tf.transpose(x, [0, 2, 1])
            x = tf.reshape(x, [B, C, H, W])
            x = self.sr(x)
            x = tf.reshape(x, [B, C, -1])
            x = tf.transpose(x, [0, 2, 1])
            x = self.norm(x)

        k = self.k(x)
        k = tf.reshape(k, [B, -1, self.num_heads, C // self.num_heads])
        k = tf.transpose(k, [0, 2, 1, 3])

        v = self.v(x)
        v = tf.reshape(v, [B, -1, self.num_heads, C // self.num_heads])
        v = tf.transpose(v, [0, 2, 1, 3])

        attn = (q @ tf.transpose(k, [0, 1, 3, 2])) * self.scale
        attn = tf.nn.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        attn = attn @ v
        attn = tf.transpose(attn, [0, 2, 1, 3])
        attn = tf.reshape(attn, shape=[B, N, C])
        x = self.proj(attn)
        x = self.proj_drop(x)
        return x
