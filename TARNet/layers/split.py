import tensorflow as tf


class Split_Streams(tf.keras.layers.Layer):
    def __init__(self):
        super(Split_Streams, self).__init__()

    def call(self, inputs):
        x, y, z = inputs

        indice_position = tf.reshape(
            tf.cast(tf.where(tf.equal(tf.reshape(y, (-1,)), z)), tf.int32),
            (-1,),
        )

        return tf.gather(x, indice_position), indice_position
