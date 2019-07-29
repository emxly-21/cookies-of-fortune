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
        A (29,)-dimensional array with each letter, delete, nothing, and space.
    """
    uppercase = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    xtrain = []
    ytrain = []

    path = "asl_alphabet_train/"
    for letter in range(26):
        l = uppercase[letter]
        ytrain.append(l)
        xtrain.append([])
        for number in range(1, 3001):
            xtrain[letter].append(cv2.imread(path + l + "/" + l + str(number) + ".jpg", cv2.IMREAD_COLOR))

    path = "asl_alphabet_train/del/"
    xtrain.append([])
    ytrain.append("del")
    for number in range(1, 3001):
        xtrain[26].append(cv2.imread(path + "del" + str(number) + ".jpg", cv2.IMREAD_COLOR))

    path = "asl_alphabet_train/nothing/"
    xtrain.append([])
    ytrain.append("nothing")
    for number in range(1, 3001):
        xtrain[27].append(cv2.imread(path + "nothing" + str(number) + ".jpg", cv2.IMREAD_COLOR))

    path = "asl_alphabet_train/space/"
    xtrain.append([])
    ytrain.append("space")
    for number in range(1, 3001):
        xtrain[28].append(cv2.imread(path + "space" + str(number) + ".jpg", cv2.IMREAD_COLOR))

    xtrain = np.array(xtrain)
    ytrain = np.array(ytrain)

    with open('xtrain.npy', mode="wb") as f:
        np.save(f, xtrain)
    with open('ytrain.npy', mode="wb") as f:
        np.save(f, ytrain)
