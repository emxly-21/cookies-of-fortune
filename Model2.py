import numpy as np
import mygrad as mg
from mygrad import Tensor

from mynn.layers.conv import conv
from mynn.layers.dense import dense
from mynn.layers.dropout import dropout
from mynn.activations.relu import relu
from mynn.initializers.glorot_uniform import glorot_uniform
from mygrad.nnet.layers import max_pool
from mynn.optimizers.adam import Adam
from mygrad.nnet.losses import softmax_crossentropy

gain = {'gain': np.sqrt(2)}
class Model:

    def __init__(self):
        self.conv1 = conv(1, 50, (5, 5),stride=5, weight_initializer=glorot_uniform, weight_kwargs=gain)
        self.conv2 = conv(50, 20, (2, 2), stride=2, weight_initializer=glorot_uniform, weight_kwargs=gain)
        self.dense1 = dense(500, 50, weight_initializer=glorot_uniform, weight_kwargs=gain)
        self.dense2 = dense(50, 29, weight_initializer=glorot_uniform, weight_kwargs=gain)

        pass

    def __call__(self, x):
#         convol1 = relu(self.conv1(x))
#         pool1 = max_pool(convol1, pool=(2,2), stride=2)
#         convol2 = relu(self.conv2(pool1))
#         pool2 = max_pool(convol2, pool=(2,2), stride=2)
#         flattened = pool2.reshape((len(x), 250))
#         den3 = relu(self.dense3(flattened))
#         den4 = self.dense4(den3)
#         return den4
        #print(self.conv1(x).shape)
        step1 = max_pool(self.conv1(x), (2, 2), stride=2)
        step2 = max_pool(self.conv2(step1), (2, 2), stride=2)
        flatten = step2.reshape(-1,500)
        dense_layers = self.dense2(relu(self.dense1(flatten)))

        return dense_layers
        # returns output of dense -> relu -> dense -> relu -> dense -> softmax three layer.

        pass

    @property
    def parameters(self):
        ''' A convenience function for getting all the parameters of our model. '''
        return self.conv1.parameters + self.conv2.parameters + self.dense1.parameters + self.dense2.parameters