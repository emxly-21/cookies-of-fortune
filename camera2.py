from camera import take_picture
import cv2
from camera import save_camera_config
import matplotlib.pyplot as plt
import numpy as np

save_camera_config(port=0, exposure=1)

def get_signs(n):
    """
    Will take in a bunch of pictures every second for a specified amount of time
    creating np.ndarrays of our signs
    Parameter: n : number of pictures desired to be taken
    Returns: np.ndarray of our image arrays
    """

    count = 0
    img_session = []
    fig, ax = plt.subplots()
    for count in range(n):
        img_array = take_picture()
        print("Picture taken")
        img_array = img_array[:, 80:560, :]
        resized = cv2.resize(img_array, (200, 200), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
        ax.imshow(gray)
        img_session.append(gray)
        plt.show()
    img_session = np.array(img_session)

    img_array = img_session.reshape(n, 40000).astype(np.float64)

    img_array -= 126.145118219
    img_array /= 52.3865033171

    return img_array


#to normalize subtract mean and divide by standard deviation