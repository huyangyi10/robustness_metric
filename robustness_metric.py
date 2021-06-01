#-*- coding = utf-8 -*-
#@Time : 2021-5-29 10:35
#@Author : CollionsHu
#@File : robustness_metric.py
#@Software : PyCharm

import os
import scipy
import random
import numpy as np
import tensorflow as tf
from setup_mnist import MNIST
from setup_cifar import CIFAR
from train_mnist import train_mnist
from train_cifar import train_cifar
from tensorflow.contrib.keras.api.keras.models import load_model
from uniform_sample_over_sphere import uniformsampleoversphere, l1_samples, l2_samples, linf_samples
from setup_imagenet import ImageNetModelPrediction, NodeLookup, create_graph, model_params, keep_aspect_ratio_transform

from tensorflow.contrib.keras.api.keras.layers import Lambda
from tensorflow.contrib.keras.api.keras.optimizers import SGD
from tensorflow.contrib.keras.api.keras.models import Sequential
from tensorflow.contrib.keras.api.keras.layers import Conv2D, MaxPooling2D
from tensorflow.contrib.keras.api.keras.layers import Dense, Dropout, Activation, Flatten
# from extreme_value_estimation import get_extreme_value_estimate
# from setup_cifar import CIFAR





#s stands for small dataset, l for large dateset
# given the test data, generate n new samples for each batch.
# given the test data, generate sample in n batch.
# the target label
# store the max_difference for each sample during the batch, like [ [1,1], [1,1] ] for two test_data & two batch

# targeted_test_sample_s stands for the number of test-set
def robustness_metric_targeted_s(model_type, targeted_test_sample_s, target_label, batch_sample_n, batch_n, norm, r):


    if not os.path.isdir('models'):
        os.makedirs('models')
    retrain_flag = False
    batch_max = []



    if model_type == "MNIST_MLP":

        # 1.train and load the model
        file_name = "models/"+model_type
        if retrain_flag: train_mnist(data = MNIST(), model_type=model_type, file_name=file_name, num_epochs=20)
        model = load_model("./"+ file_name)
        data = MNIST()

        # 2.sampling
        for i in range(targeted_test_sample_s):  # calculate the samples around the #targeted_test_sample_s test-datas

            # 2.1 get the test-data's info.
            rand_test_sample = np.random.randint(0, len(data.test_data))
            print("For the " + str(rand_test_sample) + "_th test data: ")
            test_data = data.test_data[rand_test_sample].reshape(784)  # (1, 28, 28, 1) -> (784)
            true_label = list(data.test_labels[rand_test_sample]).index(1.0)  # (10, 1) -> int
            print("The true label: ", true_label, "; The target label: ", target_label)


            # 2.2 sampling around this test-data
            batch_max_for_one = []  # store the max_difference for the sample during the batch
            for batch_i in range(batch_n):
                print("\tbatch_" + str(batch_i))
                uniformSample = uniformsampleoversphere(batch_sample_n, 784, norm, r, test_data)
                max_difference = -np.inf
                for sample_i in range(batch_sample_n):
                    check_data = uniformSample[sample_i].reshape(1, 28, 28, 1) # (784) -> (1, 28, 28, 1)
                    # ================4.prediction======================#
                    # given the data as the tuple(1, 28, 28, 1), we calculate the probability f2-f1
                    prob = model.predict(check_data)[0]
                    prob_difference = prob[target_label] - prob[true_label]
                    print("\t\tThe sample_" + str(sample_i) + "'s pro difference:", prob_difference)
                    if prob_difference > max_difference:
                        max_difference = prob_difference
                batch_max_for_one.append(max_difference)
            batch_max.append(batch_max_for_one)
            print("------------------------------------")
        print(batch_max)

    elif model_type == "CIFAR_MLP":

        # 1.train and load the model
        file_name = "models/" + model_type
        if retrain_flag: train_cifar(data=CIFAR(), model_type=model_type, file_name=file_name, num_epochs=20)
        model = load_model("./" + file_name)
        data = CIFAR()

        # 2.sampling
        for i in range(targeted_test_sample_s):  # calculate the samples around the #targeted_test_sample_s test-datas

            # 2.1 get the test-data's info.
            rand_test_sample = np.random.randint(0, len(data.test_data))
            print("For the " + str(rand_test_sample) + "_th test data: ")
            test_data = data.test_data[rand_test_sample].reshape(3072)  # (32, 32, 3) -> (3072)
            true_label = list(data.test_labels[rand_test_sample]).index(1.0)  # (10, 1) -> int
            print("The true label: ", true_label, "; The target label: ", target_label)

            # 2.2 sampling around this test-data
            batch_max_for_one = []  # store the max_difference for the sample during the batch
            for batch_i in range(batch_n):
                print("\tbatch_" + str(batch_i))
                uniformSample = uniformsampleoversphere(batch_sample_n, 3072, norm, r, test_data)
                max_difference = -np.inf
                for sample_i in range(batch_sample_n):
                    check_data = uniformSample[sample_i].reshape(1, 32, 32, 3)  # (3072) -> (32, 32, 3)
                    # ================4.prediction======================#
                    # given the data as the tuple(32, 32, 3), we calculate the probability f2-f1
                    prob = model.predict(check_data)[0]
                    prob_difference = prob[target_label] - prob[true_label]
                    print("\t\tThe sample_" + str(sample_i) + "'s pro difference:", prob_difference)
                    if prob_difference > max_difference:
                        max_difference = prob_difference
                batch_max_for_one.append(max_difference)
            batch_max.append(batch_max_for_one)
            print("------------------------------------")
        print(batch_max)

    elif model_type == "imagenet_reset":

        # 1.train and load the model
        model_name = 'resnet_v2_101'  # model_dir = 'tmp/imagenet'
        param = model_params[model_name]
        create_graph(param)
        image_size = param['size']

        # 2.sampling
        with tf.Session() as sess:
            model = ImageNetModelPrediction(sess, True, model_name)

            for i in range(targeted_test_sample_s):  # calculate the samples around the #targeted_test_sample_s test-datas

                # 2.1 get the test-data's info.
                test_data_dir = '/Users/WJ-Hong/Desktop/Research_Direction/CLEVER/minimum_adv_distortion/imagenetdata/imgs/'
                pathDir = os.listdir(test_data_dir)
                image_name = random.sample(pathDir, 1)[0]
                image = test_data_dir + image_name
                print("For the test data (" + image_name + "): ")
                dat = np.array(scipy.misc.imresize(scipy.misc.imread(image), (image_size, image_size)),
                               dtype=np.float32)
                dat /= 255.0
                dat -= 0.5
                predictions = model.predict(dat)
                # Creates node ID --> English string lookup.
                node_lookup = NodeLookup()
                true_label_prob = max(predictions)
                true_label = list(predictions).index(true_label_prob)
                if 'vgg' in model_name or 'densenet' in model_name or 'alexnet' in model_name:
                    true_label += 1
                    target_label += 1
                print("The true label: ", true_label, "; The target label: ", target_label)
                # human_string = node_lookup.id_to_string(true_label)
                # print('%s (score = %.5f)' % (human_string, true_label_prob))


                # 2.2 sampling around this test-data
                # the test-data should be the format tuple(image_size, image_size, 3), like (299, 299, 3)
                batch_max_for_one = []  # store the max_difference for the sample during the batch
                for batch_i in range(batch_n):
                    print("\tbatch_" + str(batch_i))
                    test_data = dat.reshape(image_size*image_size*3)  # (299, 299, 3) -> (268,203)
                    uniformSample = uniformsampleoversphere(batch_sample_n, image_size*image_size*3, norm, r, test_data)
                    max_difference = -np.inf
                    for sample_i in range(batch_sample_n):
                        check_data = uniformSample[sample_i].reshape(image_size, image_size, 3)  # (268,203) -> (299, 299, 3)
                        # ================4.prediction======================#
                        # given the data as the tuple(299, 299, 3), we calculate the probability f2-f1
                        predictions = model.predict(check_data)
                        prob_difference = predictions[target_label] - predictions[true_label]
                        print("\t\tThe sample_" + str(sample_i) + "'s pro difference:", prob_difference)
                        if prob_difference > max_difference:
                            max_difference = prob_difference
                    batch_max_for_one.append(max_difference)
                batch_max.append(batch_max_for_one)
                print("------------------------------------")
        print(batch_max)


    figname = ""

    # get_extreme_value_estimate(batch_max, norm, figname)





    # targeted_test_sample_s = 500
    # targeted_test_sample_l = 100
    # untargeted_test_sample_s = 100



if __name__ == '__main__':

    # supported model type: MNIST_MLP, CIFAR_MLP, imagenet_reset
    robustness_metric_targeted_s(model_type="CIFAR_MLP", targeted_test_sample_s=2, target_label=8, batch_sample_n=10, batch_n=2, norm="l2", r=1)

