from model_tester import model
import numpy as np
from camera2 import get_signs

def main():
    uppercase = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    with open('layer1_weights.npy', mode="rb") as f:
        weights1 = np.load(f)
    with open('layer1_bias.npy', mode="rb") as f:
        bias1 = np.load(f)
    with open('layer2_weights.npy', mode="rb") as f:
        weights2 = np.load(f)
    with open('layer2_bias.npy', mode="rb") as f:
        bias2 = np.load(f)
    with open('layer3_weights.npy', mode="rb") as f:
        weights3 = np.load(f)
    with open('layer3_bias.npy', mode="rb") as f:
        bias3 = np.load(f)
    with open('layer4_weights.npy', mode="rb") as f:
        weights4 = np.load(f)
    with open('layer4_bias.npy', mode="rb") as f:
        bias4 = np.load(f)

    x = get_signs(1)
    print(uppercase[np.argmax(model(x, weights1, bias1, weights2, bias2, weights3, bias3, weights4, bias4), axis=1)[0]])

if __name__ == "__main__":
    main()