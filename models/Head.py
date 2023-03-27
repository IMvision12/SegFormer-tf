import tensorflow as tf

class MLP(tf.keras.layers.Layer):
    def __init__(self, decode_dim=768, **kwargs):
        super().__init__(**kwargs)
        self.proj = tf.keras.layers.Dense(decode_dim)

    def forward(self, x):
        get_shape = tf.shape(x)
        B = get_shape[0]
        H = get_shape[1]
        W = get_shape[2]
        dim = get_shape[-1]

        x = tf.reshape(x, (B, H * W, dim))
        x = self.proj(x)
        return x


class SegFormerHead(tf.keras.layers.Layer):
    def __init__(self, num_classes, decode_dim, num_blocks=4, cls_dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)

        mlps = []
        for _ in range(num_blocks):
            mlp = MLP(decode_dim)
            mlps.append(mlp)
        self.mlps = mlps

        self.linear_fuse = tf.keras.layers.Conv2D(
            filters=decode_dim, kernel_size=1, use_bias=False
        )
        self.norm = tf.keras.layers.BatchNormalization(epsilon=1e-5)
        self.act = tf.keras.layers.Activation("relu")

        self.dropout = tf.keras.layers.Dropout(cls_dropout_rate)
        self.cls = tf.keras.layers.Conv2D(filters=num_classes, kernel_size=1)

    def call(self, x):
        get_shape = tf.shape(x[0])
        H = get_shape[1]
        W = get_shape[2]
        outputs = []
        for feat, mlp in zip(x, self.mlps):
            x = mlp(feat)
            x = tf.image.resize(x, size=(H, W), method="bilinear")
            outputs.append(x)

        x = self.linear_fuse(tf.concat(outputs[::-1], axis=3))
        x = self.norm(x)
        x = self.act(x)
        x = self.cls(x)
        return x
