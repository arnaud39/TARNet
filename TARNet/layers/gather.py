import tensorflow as tf


class Gather_Streams(tf.keras.layers.Layer):
    def __init__(self):
        super(Gather_Streams, self).__init__()

    def call(self, inputs):
        
        x, y = inputs
        return tf.dynamic_stitch(y, x)
