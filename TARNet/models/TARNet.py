import tensorflow as tf

from ..layers.gather import Gather_Streams
from ..layers.split import Split_Streams
from ..layers.MLP import MLP
from pickle import load

from typing import Any


class TARNet(tf.keras.Model):
    """Return a tarnet sub KERAS API model."""

    def __init__(
        self,
        normalizer_layer: tf.keras.layers.Layer = None,
        n_treatments: int = 2,
        output_dim: int = 1,
        phi_layers: int = 2,
        units:int = 20,
        y_layers: int = 3,
        activation: str = "relu",
        reg_l2: float = 0.0,
        treatment_as_input: bool = False,
        scaler: Any = None,
        output_bias: float = None,
    ):
        """Initialize the layers used by the model.

        Args:
            normalizer_layer (tf.keras.layer, optional): _description_. Defaults to None.
            n_treatments (int, optional): _description_. Defaults to 2.
            output_dim (int, optional): _description_. Defaults to 1.
            phi_layers (int, optional): _description_. Defaults to 2.
            y_layers (int, optional): _description_. Defaults to 3.
            activation (str, optional): _description_. Defaults to "relu".
            reg_l2 (float, optional): _description_. Defaults to 0.0.
        """
        super(TARNet, self).__init__()
        # uniform quantile transform for treatment
        self.scaler = scaler if scaler else load(open("scaler.pkl", "rb"))

        # input normalization layer
        self.normalizer_layer = normalizer_layer
        self.phi = MLP(
            units=units,
            activation=activation,
            name="phi",
            num_layers=phi_layers,
        )

        self.splitter = Split_Streams()

        self.y_hiddens = [
            MLP(
                units=units,
                activation=activation,
                name=f"y_{k}",
                num_layers=y_layers,
            )
            for k in range(n_treatments)
        ]

        # add linear function to cover the normalized output
        self.y_outputs = [
            tf.keras.layers.Dense(
                output_dim,
                activation="sigmoid",
                bias_initializer=output_bias,
                name=f"top_{k}",
            )
            for k in range(n_treatments)
        ]

        self.n_treatments = n_treatments

        self.output_ = Gather_Streams()

    def call(self, x):

        cofeatures_input, treatment_input = x
        treatment_cat = tf.cast(treatment_input, tf.int32)

        if self.normalizer_layer:
            cofeatures_input = self.normalizer_layer(cofeatures_input)
        x_flux = self.phi(cofeatures_input)

        streams = [
            self.splitter([x_flux, treatment_cat, tf.cast(indice_treatment, tf.int32)])
            for indice_treatment in range(len(self.y_hiddens))
        ]
        # xstream is a list of tuple, containing the gathered and indice position, let's unpack them
        x_streams, indice_streams = zip(*streams)
        # tf.print(indice_streams, output_stream=sys.stderr)
        x_streams = [
            y_hidden(x_stream) for y_hidden, x_stream in zip(self.y_hiddens, x_streams)
        ]
        x_streams = [
            y_output(x_stream) for y_output, x_stream in zip(self.y_outputs, x_streams)
        ]

        return self.output_([x_streams, indice_streams])
