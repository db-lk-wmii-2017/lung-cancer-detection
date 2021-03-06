#!/usr/bin/env python
# coding: utf-8

import sys
import os
import numpy as np

import tensorflow as tf

tf.get_logger().setLevel("ERROR")

from networks import CustomCNN, NetworkType
from plot_keras_history import plot_history
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.utils.vis_utils import plot_model
from utils import load_data, define_output_redirecter


DATA_PATH = os.path.join("data", "model")
MODEL_OUTPUT = os.path.join("data", "model")

if not os.path.exists(MODEL_OUTPUT):
    os.makedirs(MODEL_OUTPUT)


def define_cancer_counter(type):
    def cancer_counter(Y):
        if NetworkType.CATEGORICAL == type:
            counter = 0
            for y in Y:
                if y[0] == 0.0 and y[1] == 1.0:
                    counter += 1
            return counter
        else:
            return np.count_nonzero(Y == 1.0)

    return cancer_counter


redirect_output, restore_output = define_output_redirecter()

for type in (NetworkType.BINARY, NetworkType.CATEGORICAL):
    model_name = "CustomCNN-{}".format(type.name)
    redirect_output(os.path.join(MODEL_OUTPUT, model_name + ".log"))
    print("=== Starting {} ===".format(model_name))
    sys.stdout.flush()
    X, Y, X_test, Y_test = load_data(
        DATA_PATH,
        labels_as_categories=True if NetworkType.CATEGORICAL == type else False,
    )
    X = X / 255.0
    X_test = X_test / 255.0

    cancer_counter = define_cancer_counter(type)
    print()
    print("=== Data was load ===")
    print()
    print("Train size: {}".format(len(X)))
    print("Test size: {}".format(len(X_test)))
    print()
    print("=== Data Description ===")
    print()
    print(
        "Train:\n\t> positive: {}\n\t> negative: {}".format(
            cancer_counter(Y), len(Y) - cancer_counter(Y)
        )
    )
    print(
        "Test:\n\t> positive: {}\n\t> negative: {}".format(
            cancer_counter(Y_text), len(Y_text) - cancer_counter(Y_text)
        )
    )
    print()
    sys.stdout.flush()

    cnn = CustomCNN(model_type=type)
    cnn.define_network(X)
    cnn.summary()

    model, history = cnn.train(
        MODEL_OUTPUT,
        X,
        Y,
        X_test,
        Y_test,
        log_dir="{}-logs".format(model_name),
        checkpoint="{}.checkpoint".format(model_name),
    )
    model.save(os.path.join(MODEL_OUTPUT, model_name + ".model"))
    print("Network trained and saved as {}.model".format(model_name))
    model_json = model.to_json()
    with open(os.path.join(MODEL_OUTPUT, model_name + ".json"), "w") as json_file:
        json_file.write(model_json)
    plot_model(
        model,
        to_file=os.path.join(MODEL_OUTPUT, "{}-graph.png".format(model_name)),
        show_shapes=True,
    )
    plot_history(
        history.history,
        path=os.path.join(MODEL_OUTPUT, "{}-history.png".format(model_name)),
    )
    plt.close()
    sys.stdout.flush()

restore_output()
