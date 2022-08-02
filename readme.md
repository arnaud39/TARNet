# TARNet

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![pypi version](https://img.shields.io/pypi/v/tarnet.svg)](https://pypi.python.org/pypi/vulpes)

[![Downloads](https://static.pepy.tech/badge/tarnet)](https://pepy.tech/project/tarnet)

**TARNet: TARNet Model with tensorflow 2 API.**

Treatment-Agnostic Representation Network 🩺  is a machine learning architecture that has a common MLP feeding specific sub-networks. It can help to identify bias in the data, estimate average treatment effect or act as transfer-learning like model.

![TARNet model architecture](https://i.ibb.co/Lt7B7vV/TARNet.png)

This package implement this model as a keras-like TensorFlow API model.

Parameters are:
```python
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
```


The input wil be (X,t) with t a (X,1) shape tensor representing the hidden treatment/category.



Author & Maintainer: Arnaud Petit

## Installation

Using pip:

```python
pip install tarnet
```

## Dependencies

vulpes requires:

- Python (>= 3.7)
- TensorFlow

## Documentation

Link to the documentation: coming soon

## Examples

General case, import one of the classes Classifiers, Regressions, Clustering from vulpes.automl, add some parameters to the object (optional), fit your dataset:

```python
from tarnet import TARNet

df = pd.read_csv("...")

X, y, t = (
    df.drop(output + ["icu_type"], axis=1).to_numpy(dtype="float32"),
    df[output].to_numpy(dtype="int").reshape(-1, 1),
    df.icu_type.to_numpy(dtype="int32").reshape(-1, 1),
)

from tarNET import tarNET
import tensorflow as tf

normalizer_layer = tf.keras.layers.Normalization(axis=None)
normalizer_layer.adapt(X)
scaler = normalizer_layer

DATASET_SIZE = len(df)

batch_size = 64

train_size = int(0.7 * DATASET_SIZE)
val_size = int(0.2 * DATASET_SIZE)
test_size = int(0.1 * DATASET_SIZE)

dataset = tf.data.Dataset.zip(
    (tf.data.Dataset.from_tensor_slices((X, t)), tf.data.Dataset.from_tensor_slices(y))
).shuffle(buffer_size=DATASET_SIZE, reshuffle_each_iteration=False)#batch(64)

train_dataset = dataset.take(train_size).batch(batch_size)
test_dataset = dataset.skip(train_size)
val_dataset = test_dataset.take(val_size).batch(batch_size)
test_dataset = test_dataset.skip(val_size)

neg, pos = np.bincount(np.concatenate([y for _, y in train_dataset]).reshape(-1).astype("int"))

initial_bias = tf.keras.initializers.Constant(np.log([pos/neg]))

model = tarNET(
    output_dim=1,
    n_treatments=10,
    normalizer_layer=normalizer_layer,
    scaler=scaler,
    output_bias=initial_bias,
    phi_layers=10,
)

```


## Why TARNet?

TARNet stands for: **T**reatment-**A**gnostic **R**epresentation **N**etwork. 



## License

[MIT](https://choosealicense.com/licenses/mit/)