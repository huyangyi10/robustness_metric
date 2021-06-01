#!/usr/bin/env python
#-*- coding:utf-8 -*-

'''Trains on the CIFAR10 dataset.
CIFAR_MLP:
CIFAR_CNN:
'''

import os
import tensorflow as tf
from setup_cifar import CIFAR, CIFAR_MLP, CIFAR_CNN
from tensorflow.contrib.keras.api.keras.optimizers import RMSprop



def train_cifar(data, model_type, file_name, num_epochs=20, activation_fun="relu"):

    # ------------------------------- Parameters ----------------------------- #
    if model_type == "CIFAR_MLP":

        # network's structure
        model = CIFAR_MLP(use_softmax=True, activation=activation_fun).model

        # optimizer setting
        optimizer = RMSprop()
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])

    elif model_type == "CIFAR_CNN":

        # network's structure
        model = CIFAR_CNN(use_softmax=True, activation=activation_fun).model

        # optimizer setting
        learning_rate = 0.01
        decay = 0.5
        momentum = 0.9
        optimizer = tf.keras.optimizers.SGD(lr=learning_rate,
                                            decay=decay,
                                            momentum=momentum)
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])


    # ------------------------------- training ----------------------------- #
    # run training with given dataset, and print progress
    batch_size = 128
    epochs = num_epochs
    model.fit(data.train_data, data.train_labels,
              batch_size=batch_size,
              validation_data=(data.validation_data, data.validation_labels),
              epochs=epochs,
              verbose=2,
              shuffle=True)


    # ------------------------------- Saving ----------------------------- #
    # save model to a file
    if file_name != None:
        model.save(file_name)
    # test accuracy
    score = model.evaluate(data.test_data, data.test_labels, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    print("==========================\n\n")

    return model


###=================================================================================###
if __name__ == "__main__":
    if not os.path.isdir('models'):
        os.makedirs('models')
    train_cifar(data=CIFAR(), model_type="CIFAR_MLP", file_name="models/cifar_MLP", num_epochs=20, activation_fun="relu")
    train_cifar(data=CIFAR(), model_type="CIFAR_CNN", file_name="models/cifar_CNN", num_epochs=50, activation_fun="relu")

