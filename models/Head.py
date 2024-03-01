import keras
from keras import ops

class MLP(keras.layers.Layer):
    def __init__(self, decode_dim):
        super().__init__()
        self.proj = keras.layers.Dense(decode_dim)

    def call(self, x):
        x = self.proj(x)
        return x


class ConvModule(keras.layers.Layer):
    def __init__(self, decode_dim):
        super().__init__()
        self.conv = keras.layers.Conv2D(
            filters=decode_dim, kernel_size=1, use_bias=False
        )
        self.bn = keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.9)
        self.activate = keras.layers.ReLU()

    def call(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activate(x)
        return x


class SegFormerHead(keras.layers.Layer):
    def __init__(self, num_mlp_layers=4, decode_dim=768, num_classes=19):
        super().__init__()

        self.linear_layers = []
        for _ in range(num_mlp_layers):
            self.linear_layers.append(MLP(decode_dim))

        self.linear_fuse = ConvModule(decode_dim)
        self.dropout = keras.layers.Dropout(0.1)
        self.linear_pred = keras.layers.Conv2D(num_classes, kernel_size=1)

    def call(self, inputs):
        H = ops.shape(inputs[0])[1]
        W = ops.shape(inputs[0])[2]
        outputs = []

        for x, mlps in zip(inputs, self.linear_layers):
            x = mlps(x)
            x = ops.image.resize(x, size=(H, W), interpolation="bilinear")
            outputs.append(x)

        x = self.linear_fuse(ops.concatenate(outputs[::-1], axis=3))
        x = self.dropout(x)
        x = self.linear_pred(x)

        return x
