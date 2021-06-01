#-*- coding = utf-8 -*-
#@Time : 2021-5-29 10:35
#@Author : CollionsHu
#@File : robustness_metric.py
#@Software : PyCharm

import numpy as np
from tensorflow.contrib.keras.api.keras.models import Sequential
from tensorflow.contrib.keras.api.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.contrib.keras.api.keras.layers import Conv2D, MaxPooling2D
from tensorflow.contrib.keras.api.keras.layers import Lambda
from tensorflow.contrib.keras.api.keras.models import load_model
from tensorflow.contrib.keras.api.keras.optimizers import SGD

import tensorflow as tf
from setup_mnist import MNIST
from train_mnist import train_mnist
from uniform_sample_over_sphere import l1_samples, l2_samples, linf_samples, uniformsampleoversphere
from calculate_probability_difference import calculate
from extreme_value_estimation import get_extreme_value_estimate
from setup_imagenet import keep_aspect_ratio_transform, ImageNetModelPrediction, NodeLookup, create_graph, model_params
# from setup_cifar import CIFAR

import os

#s stands for small dataset, l for large dateset
# given the test data, generate n new samples for each batch.
# given the test data, generate sample in n batch.
# the target label
# store the max_difference for each sample during the batch, like [ [1,1], [1,1] ] for two test_data & two batch

# targeted_test_sample_s stands for the number of test-set
def robustness_metric_targeted_s(model_type, targeted_test_sample_s,target_label, batch_sample_n, batch_n, norm, r):
    if model_type == "MNIST_MLP":
        file_name = "models/"+model_type
        train_mnist(data = MNIST(), model_type=model_type, file_name=file_name, num_epochs=20)
        model = load_model("./"+ file_name)
        batch_max = []
        data = MNIST()

        for i in range(targeted_test_sample_s):  # calculate the samples around the first n test-data
            rand_test_sample = np.random.randint(0, len(data.test_data))
            print("For the " + str(rand_test_sample) + "_th test data: ")
            # print(data.test_data.shape[0])
            test_data = data.test_data[rand_test_sample].reshape(784)  # (1, 28, 28, 1) -> (784)
            true_label = list(data.test_labels[rand_test_sample]).index(1.0)  # (10, 1) -> int
            print("The true label: ", true_label, "; The target label: ", target_label)
            # generate the sampling data
            batch_max_for_one = []  # store the max_difference for the sample during the batch

            uniformSample = uniformsampleoversphere(batch_n, batch_sample_n, norm, r, test_data)

            for batch_i in range(batch_n):
                print("\tbatch_" + str(batch_i))
                max_difference = -np.inf
                for sample_i in range(batch_sample_n):
                    uniformSample = uniformsampleoversphere(784, 2, 1, test_data)  # sampling data
                    check_data = uniformSample.reshape(1, 28, 28, 1)  # (784) -> (1, 28, 28, 1)
                    # ================4.prediction======================#
                    # given the data as the tuple(1, 28, 28, 1), we calculate the probability f2-f1
                    prob_difference = calculate(model, check_data, true_label, target_label)
                    print("\t\tThe sample_" + str(sample_i) + "'s pro difference:", prob_difference)
                    if prob_difference > max_difference:
                        max_difference = prob_difference
                batch_max_for_one.append(max_difference)
            batch_max.append(batch_max_for_one)
            print("------------------------------------")
        print(batch_max)

        figname = ""

        get_extreme_value_estimate(batch_max, norm, figname)






    # targeted_test_sample_s = 500
    # targeted_test_sample_l = 100
    # untargeted_test_sample_s = 100

