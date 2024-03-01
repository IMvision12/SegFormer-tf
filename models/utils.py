import keras
from keras import ops

class ResizeLayer(keras.layers.Layer):
    def __init__(self, height, width, **kwargs):
        super(ResizeLayer, self).__init__(**kwargs)
        self.height = height
        self.width = width

    def call(self, inputs):
        resized = ops.image.resize(
            inputs,
            size=(self.height, self.width),
            interpolation="bilinear",
        )
        return resized


class DropPath(keras.layers.Layer):
    def __init__(self, drop_path, **kwargs):
        super().__init__(**kwargs)
        self.drop_path = drop_path

    def call(self, x, training=None):
        if training:
            keep_prob = 1 - self.drop_path
            shape = (ops.shape(x)[0],) + (1,) * (len(ops.shape(x)) - 1)
            random_tensor = keep_prob + keras.random.uniform(shape, 0, 1)
            random_tensor = ops.floor(random_tensor)
            return (x / keep_prob) * random_tensor
        return x
