import numpy as np
import cv2


def read_train():
    """
    Reads in training data (26 letters, delete, nothing, space) from the asl_alphabet_train folder.

    Saves
    -----
    xtrain : numpy.ndarray
        A (29, 3000, 200, 200, 3)-dimensional array with 3000 images for each letter, delete, nothing,
        and space, each with 200x200 dimensions.

    ytrain : numpy.ndarray
        A (29,)-dimensional array with each letter (numbered 0-25), delete (26), nothing (27), and space (28).
    """
    # if color images needed instead of grayscale, use cv2.IMREAD_COLOR instead of cv2.IMREAD_GRAYSCALE

    uppercase = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    xtrain = []
    ytrain = []

    path = "asl_alphabet_train/"
    for letter in range(len(uppercase)):
        for number in range(1, 1001):
            let = uppercase[letter]
            ytrain.append(letter)
            xtrain.append(cv2.imread(path + let + "/" + let + str(number) + ".jpg", cv2.IMREAD_GRAYSCALE))

    path = "asl_alphabet_train/del/"
    for number in range(1, 1001):
        ytrain.append(26)
        xtrain.append(cv2.imread(path + "del" + str(number) + ".jpg", cv2.IMREAD_GRAYSCALE))

    path = "asl_alphabet_train/nothing/"
    for number in range(1, 1001):
        ytrain.append(27)
        xtrain.append(cv2.imread(path + "nothing" + str(number) + ".jpg", cv2.IMREAD_GRAYSCALE))

    path = "asl_alphabet_train/space/"
    for number in range(1, 1001):
        ytrain.append(28)
        xtrain.append(cv2.imread(path + "space" + str(number) + ".jpg", cv2.IMREAD_GRAYSCALE))

    xtrain = np.array(xtrain).astype(np.float64)
    ytrain = np.array(ytrain)

    # for color images, reshape to 87000 x 120000
    # for all grayscale images, reshape to 29000 x 120000
    # for one-third grayscale images, reshape to 29000 x 40000
    xtrain = xtrain.reshape(29000, 40000)

    #mean_train = np.mean(xtrain)
    #sd_train = np.std(xtrain)
    #xtrain -= mean_train
    #xtrain /= sd_train

    with open('xtrain_gray_nonnorm.npy', mode="wb") as f:
        np.save(f, xtrain)
    with open('ytrain_gray_nonnorm.npy', mode="wb") as f:
        np.save(f, ytrain)


def read_test():
    uppercase = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    xtest = []
    ytest = []

    path = "asl_alphabet_test/"

    for letter in range(len(uppercase)):
        let = uppercase[letter]
        ytest.append(letter)
        xtest.append(cv2.imread(path + let + "_test.jpg", cv2.IMREAD_GRAYSCALE))

    ytest.append(26)
    xtest.append(cv2.imread(path + "del_test.jpg", cv2.IMREAD_GRAYSCALE))

    ytest.append(27)
    xtest.append(cv2.imread(path + "nothing_test.jpg", cv2.IMREAD_GRAYSCALE))

    ytest.append(28)
    xtest.append(cv2.imread(path + "space_test.jpg", cv2.IMREAD_GRAYSCALE))

    xtest = np.array(xtest)
    ytest = np.array(ytest)
    xtestlist=[[[k for k in j]for j in i] for i in xtest]
    xtest=np.array(xtestlist)
    print(xtest.shape)


    print(ytest)
    print(ytest.shape)

    xtest = xtest.reshape(29, 40000).astype(np.float64)

    mean_train = np.mean(xtest)
    sd_train = np.std(xtest)
    xtest -= mean_train
    xtest /= sd_train

    with open('xtest_gray.npy', mode="wb") as f:
        np.save(f, xtest)
    with open('ytest_gray.npy', mode="wb") as f:
        np.save(f, ytest)