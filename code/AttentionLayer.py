import tensorflow as tf

class MyLayer(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super(MyLayer, self).__init__(**kwargs)

    def build(self, proVA):
        super(MyLayer, self).build(proVA)
    def call(self, x):
        return x

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)