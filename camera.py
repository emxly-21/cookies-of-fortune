from camera import take_picture
import cv2


def get_signs():
    """
    Will take in a bunch of pictures every second for a specified amount of time
    creating np.ndarrays of our signs"""
    img_array = take_picture()
    img_array = img_array[:, 80:560, :]
    resized = cv2.resize(img_array, (200, 200), interpolation=cv2.INTER_AREA)
    ax.imshow(resized)

    return (resized)