from camera import take_picture
import cv2
from camera import save_camera_config
save_camera_config(port=0, exposure=1)

def get_signs(n):
    """
    Will take in a bunch of pictures every second for a specified amount of time
    creating np.ndarrays of our signs
    Parameter: n : number of pictures desired to be taken
    """

    count = 0
    img_session = []
    for count in range(n):
        img_array = take_picture()
        img_array = img_array[:, 80:560, :]
        resized = cv2.resize(img_array, (200, 200), interpolation=cv2.INTER_AREA)
        ax.imshow(resized)
        img_session.append(resized)
    img_session = np.array(img_session)

    img_array = img_session.reshape(n, 40000).astype(np.float64)

    mean_train = np.mean(img_array)
    sd_train = np.std(img_array)
    img_array -= mean_train
    img_array /= sd_train

    return img_array


#to normalize subtract mean and divide by standard deviation
