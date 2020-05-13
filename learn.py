#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
from networks import CustomCNN

from utils import load_data

DATA_PATH = os.path.join("data", "model")
MODEL_OUTPUT = os.path.join("data", "model18")

VERSION = "alpha0.0.1"
CLASSIFIER_NAME = "{}-classifier.tf1".format(VERSION)

X, Y, X_test, Y_text = load_data(DATA_PATH)
X = X / 255.0
X_test = X_test / 255.0

cnn = CustomCNN()
cnn.define_network(X)
model = cnn.train(MODEL_OUTPUT, X, Y, X_test, Y_text)
model.save(os.path.join(MODEL_OUTPUT, CLASSIFIER_NAME))
print("Network trained and saved as {}".format(CLASSIFIER_NAME))
