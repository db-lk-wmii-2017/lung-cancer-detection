#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
from cnn_model import CNNModel
from cnn_model_keres import CNN

from utils import TRAIN_DATA_FILE_NAME, TEST_DATA_FILE_NAME
import tflearn

DATA_PATH = os.path.join('data', 'model')
MODEL_OUTPUT = os.path.join('data', 'model18')

VERSION = "alpha0.0.1"
CLASSIFIER_NAME = "{}-classifier.tf1".format(VERSION)


def get_data_from_file(path):
    data = []
    labels = []
    with open(path) as file:
        while True:
            line = file.readline()
            if not line:
                break

            sample_path, label = line.split(' ')
            array = np.load(sample_path)
            if array.shape == (50, 50):
                data.append(array)
                labels.append([0.0, 1.0] if int(label) else [1.0, 0.0])
            else:
                print("Shape error: {}".format(sample_path))
    return np.asarray(data, dtype='f').reshape(
        [-1, 50, 50, 1]) / 255.0, np.asarray(labels, dtype='f')


X, Y = get_data_from_file(os.path.join(DATA_PATH, TRAIN_DATA_FILE_NAME))
X_test, Y_text = get_data_from_file(
    os.path.join(DATA_PATH, TEST_DATA_FILE_NAME))

cnn = CNN()
cnn.define_network(X)
model = cnn.train(MODEL_OUTPUT, X, Y, X_test, Y_text)
model.save(os.path.join(MODEL_OUTPUT, CLASSIFIER_NAME))
print("Network trained and saved as {}".format(CLASSIFIER_NAME))
# convnet = CNNModel()
# network = convnet.define_network(X)
# model = tflearn.DNN(network,
#                     tensorboard_verbose=0,
#                     checkpoint_path=os.path.join(MODEL_OUTPUT,
#                                                  CLASSIFIER_NAME))
# model.fit(X,
#           Y,
#           n_epoch=70,
#           shuffle=True,
#           validation_set=(X_test, Y_text),
#           show_metric=True,
#           batch_size=96,
#           snapshot_epoch=True,
#           run_id=VERSION)
# model.save(os.path.join(MODEL_OUTPUT, CLASSIFIER_NAME))
# print("Network trained and saved as {}".format(CLASSIFIER_NAME))
