#-*- coding = utf-8 -*-
#@Time : 2021-5-29 8:41
#@Author : CollionsHu
#@File : setup_mnist.py
#@Software : PyCharm

import os
import gzip
import numpy as np
import urllib.request
from tensorflow.contrib.keras.api.keras import backend as K
from tensorflow.contrib.keras.api.keras.models import Sequential
from tensorflow.contrib.keras.api.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, Dropout


def extract_data(filename, num_images):
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(num_images * 28 * 28)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = (data / 255) - 0.5  # normalize to [-0.5, 0.5]
        data = data.reshape(num_images, 28, 28, 1)
        return data


def extract_labels(filename, num_images):
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8)
    return (np.arange(10) == labels[:, None]).astype(np.float32)


class MNIST:
    def __init__(self):
        if not os.path.exists("data"):
            os.mkdir("data")
            files = ["train-images-idx3-ubyte.gz",
                     "t10k-images-idx3-ubyte.gz",
                     "train-labels-idx1-ubyte.gz",
                     "t10k-labels-idx1-ubyte.gz"]
            for name in files:
                urllib.request.urlretrieve('http://yann.lecun.com/exdb/mnist/' + name, "data/" + name)

        train_data = extract_data("data/train-images-idx3-ubyte.gz", 60000)
        train_labels = extract_labels("data/train-labels-idx1-ubyte.gz", 60000)
        self.test_data = extract_data("data/t10k-images-idx3-ubyte.gz", 10000)
        self.test_labels = extract_labels("data/t10k-labels-idx1-ubyte.gz", 10000)

        VALIDATION_SIZE = 5000

        self.validation_data = train_data[:VALIDATION_SIZE, :, :, :]
        self.validation_labels = train_labels[:VALIDATION_SIZE]
        self.train_data = train_data[VALIDATION_SIZE:, :, :, :]
        self.train_labels = train_labels[VALIDATION_SIZE:]


class MNIST_MLP:
    def __init__(self, use_softmax=True, activation="relu"):
        def bounded_relu(x):
            return K.relu(x, max_value=1)

        if activation == "brelu":
            activation = bounded_relu

        model = Sequential()
        model.add(Flatten(input_shape=(28, 28, 1)))
        model.add(Dense(200))
        model.add(Activation(activation))
        model.add(Dropout(0.5))
        model.add(Dense(10))
        if use_softmax:
            model.add(Activation('softmax'))
        model.summary()

        self.num_channels = 1
        self.image_size = 28
        self.num_labels = 10
        self.model = model


class MNIST_CNN:
    def __init__(self, use_softmax=True, activation="relu"):
        def bounded_relu(x):
            return K.relu(x, max_value=1)

        if activation == "brelu":
            activation = bounded_relu

        model = Sequential()
        model.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1)))
        model.add(Activation(activation))
        model.add(Dropout(0.5))
        model.add(Conv2D(32, (3, 3)))
        model.add(Activation(activation))
        model.add(Dropout(0.5))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))

        model.add(Conv2D(64, (3, 3)))
        model.add(Activation(activation))
        model.add(Dropout(0.5))
        model.add(Conv2D(64, (3, 3)))
        model.add(Activation(activation))
        model.add(Dropout(0.5))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(200))
        model.add(Activation(activation))
        model.add(Dropout(0.5))
        model.add(Dense(200))
        model.add(Activation(activation))
        model.add(Dropout(0.5))
        model.add(Dense(10))
        if use_softmax:
            model.add(Activation('softmax'))
        model.summary()

        self.num_channels = 1
        self.image_size = 28
        self.num_labels = 10
        self.model = model
