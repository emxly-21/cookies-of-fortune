import numpy as np
import mygrad as mg
from mygrad import Tensor

from mynn.layers.conv import conv
from mynn.layers.dense import dense
from mynn.layers.dropout import dropout
from mynn.activations.relu import relu
from mynn.initializers.glorot_uniform import glorot_uniform
from mygrad.nnet.layers import max_pool
from mynn.optimizers.sgd import SGD
from mygrad.nnet.losses import softmax_crossentropy


class Model:

    def __init__(self):
        self.conv1 = conv(1, 50, 3, 3, stride=1, weight_initializer=glorot_uniform)
        self.conv2 = conv(50, 20, 3, 3, stride=1, weight_initializer=glorot_uniform)
        self.dense1 = dense(180, 50, weight_initializer=glorot_uniform)
        self.dense2 = dense(50, 29, weight_initializer=glorot_uniform)

        pass

    def __call__(self, x):
        step1 = max_pool(relu(self.conv1(x)), (2, 2), stride=2)
        step2 = max_pool(relu(self.conv2(step1)), (2, 2), stride=2)

        flatten = step2.reshape(len(x), )
        dense_layers = self.dense2(relu(self.dense1(flatten)))

        return dense_layers
        # returns output of dense -> relu -> dense -> relu -> dense -> softmax three layer.

        pass

    @property
    def parameters(self):
        ''' A convenience function for getting all the parameters of our model. '''
        return self.conv1.parameters + self.conv2.parameters + self.dense1.parameters + self.dense2.parameters