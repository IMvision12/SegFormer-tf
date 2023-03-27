import tensorflow as tf


class MLP(tf.keras.layers.Layer):
    def __init__(self, decode_dim):
        super().__init__()
        self.proj = tf.keras.layers.Dense(decode_dim)

    def call(self, x):
        x = self.proj(x)
        return x


class ConvModule(tf.keras.layers.Layer):
    def __init__(self, decode_dim):
        super().__init__()
        self.conv = tf.keras.layers.Conv2D(
            filters=decode_dim, kernel_size=1, use_bias=False
        )
        self.bn = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.9)
        self.activate = tf.keras.layers.ReLU()

    def call(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activate(x)
        return x


class SegFormerHead(tf.keras.layers.Layer):
    def __init__(self, num_mlp_layers=4, decode_dim=768, num_classes=19):
        super().__init__()

        self.linear_layers = []
        for _ in range(num_mlp_layers):
            self.linear_layers.append(MLP(decode_dim))

        self.linear_fuse = ConvModule(decode_dim)
        self.dropout = tf.keras.layers.Dropout(0.1)
        self.linear_pred = tf.keras.layers.Conv2D(num_classes, kernel_size=1)

    def call(self, inputs):
        H = tf.shape(inputs[0])[1]
        W = tf.shape(inputs[0])[2]
        outputs = []

        for x, mlps in zip(inputs, self.linear_layers):
            x = mlps(x)
            x = tf.image.resize(x, size=(H, W), method="bilinear")
            outputs.append(x)

        x = self.linear_fuse(tf.concat(outputs[::-1], axis=3))
        x = self.dropout(x)
        x = self.linear_pred(x)

        return x
