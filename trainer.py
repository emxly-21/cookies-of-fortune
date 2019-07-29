import Model
from mynn.optimizers.sgd import SGD
from mygrad.nnet.losses import softmax_crossentropy


#needs import/download of dataset
#our train = x_train
#our truth = y_train


model = Model()
lr = #enter learning rate
mn= # put in momentum
wd = # put in weight decay

optimization = SGD(model.parameters, learning_rate=lr, momentum=mn, weight_decay=wd)

batch_size = #give number here

for batch_cnt in range(0, len(x_train) // batch_size):
    batch_indices = idxs[batch_cnt * batch_size: (batch_cnt + 1) * batch_size]
    batch = x_train[batch_indices]  # random batch of our training data

    # compute the predictions for this batch by calling on model
    predictions = model(batch)
    pass

    truth = y_train[batch_indices]
    pass

    # compute the loss
    loss = softmax_crossentropy(predictions, truth)
    pass

    # back-propagate through your computational graph through your loss
    loss.backward()
    pass

    # compute the accuracy between the prediction and the truth
    acc = accuracy(pred, truth)
    pass

    # execute gradient descent by calling step() of optim
    optimization.step()
    pass

    # null your gradients
    loss.null_gradients()