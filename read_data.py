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
    uppercase = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    xtrain = []
    ytrain = []

    path = "asl_alphabet_train/"
    for letter in range(len(uppercase)):
        for number in range(1, 3001):
            let = uppercase[letter]
            ytrain.append(letter)
            xtrain.append(cv2.imread(path + let + "/" + let + str(number) + ".jpg", cv2.IMREAD_GRAYSCALE))

    path = "asl_alphabet_train/del/"
    for number in range(1, 3001):
        ytrain.append(26)
        xtrain.append(cv2.imread(path + "del" + str(number) + ".jpg", cv2.IMREAD_GRAYSCALE))

    path = "asl_alphabet_train/nothing/"
    for number in range(1, 3001):
        ytrain.append(27)
        xtrain.append(cv2.imread(path + "nothing" + str(number) + ".jpg", cv2.IMREAD_GRAYSCALE))

    path = "asl_alphabet_train/space/"
    for number in range(1, 3001):
        ytrain.append(28)
        xtrain.append(cv2.imread(path + "space" + str(number) + ".jpg", cv2.IMREAD_GRAYSCALE))

    xtrain = np.array(xtrain)
    ytrain = np.array(ytrain)

    xtrain = xtrain.reshape(87000, 120000)

    with open('xtrain_gray.npy', mode="wb") as f:
        np.save(f, xtrain)
    with open('ytrain_gray.npy', mode="wb") as f:
        np.save(f, ytrain)


#from skimage import io
#img = io.imread('image.png', as_grey=True)
#or
from skimage import color
from skimage import io

#img = color.rgb2gray(io.imread('image.png'))

#img = cv2.imread('example.jpg', 0)