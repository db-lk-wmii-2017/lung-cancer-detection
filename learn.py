#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
from cnn import CNN

from utils import TRAIN_DATA_FILE_NAME, TEST_DATA_FILE_NAME

DATA_PATH = os.path.join("data", "model")
MODEL_OUTPUT = os.path.join("data", "model18")

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

            sample_path, label = line.split(" ")
            array = np.load(sample_path)
            if array.shape == (50, 50):
                data.append(array)
                labels.append([0.0, 1.0] if int(label) else [1.0, 0.0])
            else:
                print("Shape error: {}".format(sample_path))
    return (
        np.asarray(data, dtype="f").reshape([-1, 50, 50, 1]) / 255.0,
        np.asarray(labels, dtype="f"),
    )


X, Y = get_data_from_file(os.path.join(DATA_PATH, TRAIN_DATA_FILE_NAME))
X_test, Y_text = get_data_from_file(os.path.join(DATA_PATH, TEST_DATA_FILE_NAME))

cnn = CNN()
cnn.define_network(X)
model = cnn.train(MODEL_OUTPUT, X, Y, X_test, Y_text)
model.save(os.path.join(MODEL_OUTPUT, CLASSIFIER_NAME))
print("Network trained and saved as {}".format(CLASSIFIER_NAME))
