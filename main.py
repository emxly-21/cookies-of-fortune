from read_data import read_train
import numpy as np

def main():
    with open('xtrain_gray_nonnorm.npy', mode="rb") as f:
        xtrain = np.load(f)

    print("mean", np.mean(xtrain))
    print("std", np.std(xtrain))

if __name__ == "__main__":
    main()