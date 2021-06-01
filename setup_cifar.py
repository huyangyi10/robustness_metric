#!/usr/bin/env python
#-*- coding:utf-8 -*-


## setup_cifar.py -- cifar data and model loading code
import os
import pickle
import numpy as np
import urllib.request

from tensorflow.contrib.keras.api.keras import backend as K
from tensorflow.contrib.keras.api.keras.models import Sequential
from tensorflow.contrib.keras.api.keras.layers import Conv2D, MaxPooling2D
from tensorflow.contrib.keras.api.keras.layers import Dense, Activation, Flatten, Dropout


def load_batch(fpath, label_key='labels'):
    f = open(fpath, 'rb')
    d = pickle.load(f, encoding="bytes")
    for k, v in d.items():
        del(d[k])
        d[k.decode("utf8")] = v
    f.close()
    data = d["data"]
    labels = d[label_key]

    data = data.reshape(data.shape[0], 3, 32, 32)
    final = np.zeros((data.shape[0], 32, 32, 3),dtype=np.float32)
    final[:,:,:,0] = data[:,0,:,:]
    final[:,:,:,1] = data[:,1,:,:]
    final[:,:,:,2] = data[:,2,:,:]

    final /= 255
    final -= .5
    labels2 = np.zeros((len(labels), 10))
    labels2[np.arange(len(labels2)), labels] = 1

    return final, labels

def load_batch(fpath):
    f = open(fpath,"rb").read()
    size = 32*32*3+1
    labels = []
    images = []
    for i in range(10000):
        arr = np.fromstring(f[i*size:(i+1)*size],dtype=np.uint8)
        lab = np.identity(10)[arr[0]]
        img = arr[1:].reshape((3,32,32)).transpose((1,2,0))

        labels.append(lab)
        images.append((img/255)-.5)
    return np.array(images),np.array(labels)
    

class CIFAR:
    def __init__(self):
        train_data = []
        train_labels = []
        
        if not os.path.exists("cifar-10-batches-bin"):
            urllib.request.urlretrieve("https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz",
                                       "cifar-data.tar.gz")
            os.popen("tar -xzf cifar-data.tar.gz").read()
            

        for i in range(5):
            r,s = load_batch("cifar-10-batches-bin/data_batch_"+str(i+1)+".bin")
            train_data.extend(r)
            train_labels.extend(s)
            
        train_data = np.array(train_data,dtype=np.float32)
        train_labels = np.array(train_labels)
        
        self.test_data, self.test_labels = load_batch("cifar-10-batches-bin/test_batch.bin")
        
        VALIDATION_SIZE = 5000
        
        self.validation_data = train_data[:VALIDATION_SIZE, :, :, :]
        self.validation_labels = train_labels[:VALIDATION_SIZE]
        self.train_data = train_data[VALIDATION_SIZE:, :, :, :]
        self.train_labels = train_labels[VALIDATION_SIZE:]


class CIFAR_MLP:
    def __init__(self, use_softmax=True, activation = "relu"):
        def bounded_relu(x):
            return K.relu(x, max_value=1)
        if activation == "brelu":
            activation = bounded_relu

        model = Sequential()
        model.add(Flatten(input_shape=(32, 32, 3)))
        model.add(Dense(1024))
        model.add(Activation(activation))
        model.add(Dropout(0.5))
        model.add(Dense(10))
        if use_softmax:
            model.add(Activation('softmax'))
        model.summary()


        self.num_channels = 3
        self.image_size = 32
        self.num_labels = 10
        self.model = model



class CIFAR_CNN:
    def __init__(self, use_softmax=True, activation = "relu"):
        def bounded_relu(x):
            return K.relu(x, max_value=1)
        if activation == "brelu":
            activation = bounded_relu


        model = Sequential()
        model.add(Conv2D(64, (3, 3), input_shape=(32, 32, 3)))
        model.add(Activation(activation))
        model.add(Dropout(0.5))
        model.add(Conv2D(64, (3, 3)))
        model.add(Activation(activation))
        model.add(Dropout(0.5))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))
        
        model.add(Conv2D(128, (3, 3)))
        model.add(Activation(activation))
        model.add(Dropout(0.5))
        model.add(Conv2D(128, (3, 3)))
        model.add(Activation(activation))
        model.add(Dropout(0.5))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))
        
        model.add(Flatten())
        model.add(Dense(256))
        model.add(Activation(activation))
        model.add(Dropout(0.5))
        model.add(Dense(256))
        model.add(Activation(activation))
        model.add(Dropout(0.5))
        model.add(Dense(10))
        if use_softmax:
            model.add(Activation('softmax'))


        self.num_channels = 3
        self.image_size = 32
        self.num_labels = 10
        self.model = model
        

