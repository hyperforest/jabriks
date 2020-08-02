from jabriks.utils import to_categorical
from jabriks.models import CClassifier
from jabriks.layers import CInput, CDense
from jabriks.optimizers import SGD

import pandas as pd

# try model on Iris dataset
df = pd.read_csv('iris.csv')
X = df.iloc[:, :4].values
y = df.iloc[:, -1].values - 1
y = y.reshape(-1, 1)
y = to_categorical(y, 3)

model = CClassifier(phase_encoder='amin_murase', seed=42)
model.add(CInput(X.shape[-1]))
model.add(CDense(3, activations='amin_murase2'))
model.build(loss='sum_squared_error', optimizer=SGD(lr=0.7))
model.summary()

epoch_index, hist = model.fit(X, y, epochs=10, batch_size=1)

# try model on XOR problem
import numpy as np
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype='float')
y = np.array([[1, 0], [0, 1], [0, 1], [1, 0]], dtype='float')

model = CClassifier(phase_encoder='erlangga', seed=42)
model.add(CInput(X.shape[-1]))
model.add(CDense(2, activations='amin_murase2'))
model.build(loss='sum_squared_error', optimizer=SGD(lr=0.7))

epoch_index, hist = model.fit(X, y, epochs=25, batch_size=1)