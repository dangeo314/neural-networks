import matplotlib.pyplot as plt
import struct
import numpy as np

def unpack_labels(path):
    with open(path, 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
    return data

def unpack_imgs(path): #outputs a numpy array with shape (60000,784)
    with open(path, 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        rows, cols = struct.unpack(">II", f.read(8))
        data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        data = data.reshape((size, rows * cols))

    return data


def load_data():  # returns a tuple containing train data and test data
    train_labels = unpack_labels("../data/train-labels-idx1-ubyte")
    test_labels = unpack_labels("../data/t10k-labels-idx1-ubyte")
    train_imgs = unpack_imgs("../data/train-images-idx3-ubyte")/255
    test_imgs = unpack_imgs("../data/t10k-images-idx3-ubyte")/255

    train_data = [(i.reshape((784,1)), vectorize(l)) for (i,l) in zip(train_imgs, train_labels)]
    test_data = [(i.reshape((784,1)), l) for (i,l) in zip(test_imgs, test_labels)]
    return (train_data, test_data)


def vectorize(label):
    vector = np.zeros((10,1))
    vector[label][0] = 1;
    return vector








