import numpy as np
import mygrad as mg
import Model2
from mygrad import Tensor
import cv2

from noggin import create_plot
import matplotlib.pyplot as plt

from Model import Model
from mynn.layers.conv import conv
from mynn.layers.dense import dense
from mynn.layers.dropout import dropout
from mynn.activations.relu import relu
from mynn.initializers.glorot_uniform import glorot_uniform
from mygrad.nnet.layers import max_pool
from mynn.optimizers.sgd import SGD
from mygrad.nnet.losses import softmax_crossentropy


def accuracy(predictions, truth):
    """
    Returns the mean classification accuracy for a batch of predictions.

    Parameters
    ----------
    predictions : Union[numpy.ndarray, mg.Tensor], shape=(M, D)
        The scores for D classes, for a batch of M data points
    truth : numpy.ndarray, shape=(M,)
        The true labels for each datum in the batch: each label is an
        integer in [0, D)

    Returns
    -------
    float
    """
    if isinstance(predictions, mg.Tensor):
        predictions = predictions.data
    return np.mean(np.argmax(predictions, axis=1) == truth)

with open('xtrain_gray.npy', mode="rb") as f:
    xtrain = np.load(f)
with open('ytrain_gray.npy', mode="rb") as f:
    ytrain = np.load(f)

idxs = np.arange(len(xtrain))  # -> array([0, 1, ..., 9999])
np.random.shuffle(idxs)
x_train = []
y_train = []
for i in idxs:
    x_train.append(xtrain[i])
    y_train.append(ytrain[i])
x_train = np.array(x_train)
y_train = np.array(y_train)

model = Model2.Model()
lr = 0.01
mn = 0.9
wd = 5e-04

optimization = Adam(model.parameters, learning_rate=lr)

batch_size = 1000
idxs = np.arange(len(xtrain))  # -> array([0, 1, ..., 9999])
np.random.shuffle(idxs)

for batch_cnt in range(0, len(xtrain) // batch_size):
    batch_indices = idxs[batch_cnt * batch_size: (batch_cnt + 1) * batch_size]
    batch = xtrain[batch_indices]  # random batch of our training data

    # compute the predictions for this batch by calling on model
    # print(batch.shape)
    batch = batch.reshape(-1, 1, 200, 200)
    predictions = model(batch)
    truth = ytrain[batch_indices]

    # compute the loss
    loss = softmax_crossentropy(predictions, truth)
    acc = accuracy(predictions, truth)

    # back-propagate through your computational graph through your loss
    loss.backward()
    # compute the accuracy between the prediction and the truth
    # acc = accuracy(predictions, truth)

    # execute gradient descent by calling step() of optimization
    optimization.step()

    # null your gradients
    loss.null_gradients()

    plotter.set_test_batch({"loss": loss.item(), "accuracy": acc}, batch_size=batch_size)
    plotter.set_test_epoch()