import keras
from keras import ops
import math


class Attention(keras.layers.Layer):
    def __init__(
        self,
        dim,
        num_heads,
        sr_ratio,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = self.dim // self.num_heads

        self.units = self.num_heads * self.head_dim
        self.sqrt_of_units = math.sqrt(self.head_dim)

        self.q = keras.layers.Dense(self.units)
        self.k = keras.layers.Dense(self.units)
        self.v = keras.layers.Dense(self.units)

        self.attn_drop = keras.layers.Dropout(attn_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = keras.layers.Conv2D(
                filters=dim, kernel_size=sr_ratio, strides=sr_ratio, name='sr',
            )
            self.norm = keras.layers.LayerNormalization(epsilon=1e-05)
           
        self.proj = keras.layers.Dense(dim)
        self.proj_drop = keras.layers.Dropout(proj_drop)

    def call(
        self,
        x,
        H,
        W,
    ):
        get_shape = ops.shape(x)
        B = get_shape[0]
        C = get_shape[2]

        q = self.q(x)
        q = ops.reshape(
            q, (ops.shape(q)[0], -1, self.num_heads, self.head_dim)
        )
        q = ops.transpose(q, axes=[0, 2, 1, 3])

        if self.sr_ratio > 1:
            x = ops.reshape(x, (B, H, W, C))
            x = self.sr(x)
            x = ops.reshape(x, (B, -1, C))
            x = self.norm(x)

        k = self.k(x)
        k = ops.reshape(
            k, (ops.shape(k)[0], -1, self.num_heads, self.head_dim)
        )
        k = ops.transpose(k, axes=[0, 2, 1, 3])

        v = self.v(x)
        v = ops.reshape(
            v, (ops.shape(v)[0], -1, self.num_heads, self.head_dim)
        )
        v = ops.transpose(v, axes=[0, 2, 1, 3])

        attn = ops.matmul(q, k)
        scale = ops.cast(self.sqrt_of_units, dtype=attn.dtype)
        attn = ops.divide(attn, scale)

        attn = ops.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)
        x = ops.matmul(attn, v)
        x = ops.transpose(x, axes=[0, 2, 1, 3])
        x = ops.reshape(x, (B, -1, self.units))
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
