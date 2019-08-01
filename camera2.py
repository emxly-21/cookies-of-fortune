from camera import take_picture
import cv2
from camera import save_camera_config
import matplotlib.pyplot as plt
import numpy as np
from model_tester import model

save_camera_config(port=0, exposure=1)

def get_signs(n, weights1, bias1, weights2, bias2, weights3, bias3, weights4, bias4):
    """
    Will take in a bunch of pictures every second for a specified amount of time
    creating np.ndarrays of our signs
    Parameter: n : number of pictures desired to be taken
    Returns: np.ndarray of our image arrays
    """

    uppercase = "ABCDEFGHIJKLMNOPQRSTUVWXYZ!? "
    str = ""
    img_session = []
    fig, ax = plt.subplots()
    for cnt in range(n):
        img_array = take_picture()
        print("Picture taken")
        img_array = img_array[:, 280:1000, :]
        resized = cv2.resize(img_array, (200, 200), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
        ax.imshow(gray, cmap = plt.cm.gray)
        plt.show()
        gray -= 126.145118219
        gray /= 52.3865033171
        str += uppercase[np.argmax(model(gray, weights1, bias1, weights2, bias2, weights3, bias3, weights4, bias4), axis=1)[0]]

    return str


#to normalize subtract mean and divide by standard deviation