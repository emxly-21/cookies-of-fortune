from Model import Model
from mynn.layers.conv import conv
from mynn.layers.dense import dense
from mynn.layers.dropout import dropout
from mynn.activations.relu import relu
from mynn.initializers.glorot_uniform import glorot_uniform
from mygrad.nnet.layers import max_pool
from mynn.optimizers.sgd import SGD
from mygrad.nnet.losses import softmax_crossentropy



#needs import/download of data set
#our train = xtrain
#our truth = ytrain


model = Model()
lr = 0.01
mn= 0.9
wd = 5e-04

optimization = SGD(model.parameters, learning_rate=lr, momentum=mn, weight_decay=wd)

batch_size = #give number here

for batch_cnt in range(0, len(xtrain) // batch_size):
    batch_indices = idxs[batch_cnt * batch_size: (batch_cnt + 1) * batch_size]
    batch = xtrain[batch_indices]  # random batch of our training data

    # compute the predictions for this batch by calling on model
    predictions = model(batch)
    pass

    truth = ytrain[batch_indices]
    pass

    # compute the loss
    loss = softmax_crossentropy(predictions, truth)
    pass

    # back-propagate through your computational graph through your loss
    loss.backward()
    pass

    # compute the accuracy between the prediction and the truth
    #acc = accuracy(predictions, truth)
    pass

    # execute gradient descent by calling step() of optimization
    optimization.step()
    pass

    # null your gradients
    loss.null_gradients()