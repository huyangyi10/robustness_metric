#-*- coding = utf-8 -*-
#@Time : 2021-5-29 11:28
#@Author : CollionsHu
#@File : train_mnist.py
#@Software : PyCharm


'''Trains on the MNIST dataset.
MNIST_MLP: 97.73%
MNIST_CNN: 10.09%
'''

import os
import tensorflow as tf
from setup_mnist import MNIST, MNIST_MLP, MNIST_CNN
from tensorflow.contrib.keras.api.keras.optimizers import RMSprop




def train_mnist(data, model_type, file_name, num_epochs=20, activation_fun="relu"):

    # ------------------------------- Parameters ----------------------------- #
    if model_type == "MNIST_MLP":

        # network's structure
        model = MNIST_MLP(use_softmax=True, activation=activation_fun).model

        # optimizer setting
        optimizer = RMSprop()
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])

    elif model_type == "MNIST_CNN":

        # network's structure
        model = MNIST_CNN(use_softmax=True, activation=activation_fun).model

        # optimizer setting
        learning_rate = 0.1
        momentum = 0.9
        dropout = 0.5
        optimizer = tf.keras.optimizers.SGD(lr=learning_rate,
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