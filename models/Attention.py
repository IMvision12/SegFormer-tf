import tensorflow as tf
import math


class Attention(tf.keras.layers.Layer):
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

        self.q = tf.keras.layers.Dense(self.units)
        self.k = tf.keras.layers.Dense(self.units)
        self.v = tf.keras.layers.Dense(self.units)

        self.attn_drop = tf.keras.layers.Dropout(attn_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = tf.keras.layers.Conv2D(
                filters=dim, kernel_size=sr_ratio, strides=sr_ratio, name='sr',
            )
            self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-05)
           
        self.proj = tf.keras.layers.Dense(dim)
        self.proj_drop = tf.keras.layers.Dropout(proj_drop)

    def call(
        self,
        x,
        H,
        W,
    ):
        get_shape = tf.shape(x)
        B = get_shape[0]
        C = get_shape[2]

        q = self.q(x)
        q = tf.reshape(
            q, shape=(tf.shape(q)[0], -1, self.num_heads, self.head_dim)
        )
        q = tf.transpose(q, perm=[0, 2, 1, 3])

        if self.sr_ratio > 1:
            x = tf.reshape(x, (B, H, W, C))
            x = self.sr(x)
            x = tf.reshape(x, (B, -1, C))
            x = self.norm(x)

        k = self.k(x)
        k = tf.reshape(
            k, shape=(tf.shape(k)[0], -1, self.num_heads, self.head_dim)
        )
        k = tf.transpose(k, perm=[0, 2, 1, 3])

        v = self.v(x)
        v = tf.reshape(
            v, shape=(tf.shape(v)[0], -1, self.num_heads, self.head_dim)
        )
        v = tf.transpose(v, perm=[0, 2, 1, 3])

        attn = tf.matmul(q, k, transpose_b=True)
        scale = tf.cast(self.sqrt_of_units, dtype=attn.dtype)
        attn = tf.divide(attn, scale)

        attn = tf.nn.softmax(logits=attn, axis=-1)
        attn = self.attn_drop(attn)
        x = tf.matmul(attn, v)
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        x = tf.reshape(x, (B, -1, self.units))
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
