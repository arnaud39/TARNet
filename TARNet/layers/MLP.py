import tensorflow as tf

from tensorflow.keras import regularizers


class MLP(tf.keras.layers.Layer):
    """Multi Layer Perceptron as a Keras layer."""

    def __init__(
        self,
        units: int = 200,
        num_layers: int = 1,
        kernel_initializer=tf.keras.initializers.HeNormal(),
        activation: str = "relu",
        name: str = "phi",
        reg_l2: float = 0,
    ):
        """Initiate the layers used by the multi layer perceptron.

        Args:
            units (int, optional): _description_. Defaults to 200.
            num_layers (int, optional): _description_. Defaults to 1.
            kernel_initializer (_type_, optional): _description_. Defaults to tf.keras.initializers.HeNormal().
            activation (str, optional): _description_. Defaults to "relu".
            name (str, optional): _description_. Defaults to "phi".
            reg_l2 (float, optional): _description_. Defaults to 0.
        """

        super(MLP, self).__init__()
        self.layers = [
            tf.keras.layers.Dense(
                units=units,
                activation=activation,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=regularizers.l2(reg_l2),
                name=f"{name}_{k}",
                trainable=True,
            )
            for k in range(num_layers)
        ]

    def call(self, x: tf.Tensor) -> tf.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
