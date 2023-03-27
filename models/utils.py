import tensorflow as tf


class ResizeLayer(tf.keras.layers.Layer):
    def __init__(self, height, width, **kwargs):
        super(ResizeLayer, self).__init__(**kwargs)
        self.height = height
        self.width = width

    def call(self, inputs):
        resized = tf.image.resize(
            inputs,
            size=(self.height, self.width),
            method=tf.image.ResizeMethod.BILINEAR,
        )
        return resized


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
