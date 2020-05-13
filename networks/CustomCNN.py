from tensorflow import keras
from keras import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import BatchNormalization
from keras.layers import (
    Conv2D,
    Flatten,
    MaxPooling2D,
    Dropout,
    BatchNormalization,
    Dense,
    InputLayer,
)
from keras import regularizers
from keras.datasets import cifar10
from keras import activations
from keras.optimizers import Adadelta
from keras import metrics
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
import random
import numpy as np
import scipy
import os
from preprocess import apply_gaussian_filter


class CustomCNN(object):
    def __init__(self):
        self.model = Sequential()
        self.datagen = None
        self.validgen = None

    def create_conv_layer(
        self,
        name,
        filters,
        kernel_size=(5, 5),
        activation=activations.relu,
        regularizer=regularizers.l2(),
        padding="same",
    ):
        return Conv2D(
            filters,
            name=name,
            kernel_size=kernel_size,
            activation=activation,
            kernel_regularizer=regularizer,
            bias_regularizer=regularizer,
            padding=padding,
        )

    def create_mp_layer(self, name, pool_size=(2, 2), padding="same"):
        return MaxPooling2D(name=name, pool_size=pool_size, padding=padding)

    def init_data_generator(self, X):
        self.datagen = ImageDataGenerator(
            featurewise_center=True,
            horizontal_flip=True,
            rotation_range=25,
            featurewise_std_normalization=True,
            preprocessing_function=apply_gaussian_filter,
        )

        self.validgen = ImageDataGenerator(
            featurewise_center=True, featurewise_std_normalization=True,
        )

        self.datagen.fit(X)
        self.validgen.fit(X)

    def define_network(self, X, optimizer=Adam(learning_rate=0.0001)):
        self.init_data_generator(X)
        layers = (
            InputLayer(name="input", input_shape=(X.shape[1], X.shape[2], X.shape[3])),
            self.create_conv_layer("conv_1_32", 32),
            self.create_mp_layer("mp1"),
            self.create_conv_layer("conv_2_64", 64),
            self.create_conv_layer("conv_3_64", 64, kernel_size=(3, 3)),
            self.create_mp_layer("mp2"),
            Flatten(),
            Dense(512, activation=activations.relu, name="dense1"),
            Dropout(0.5, name="dp1"),
            Dense(2, activation=activations.softmax, name="dense2"),
        )

        for layer in layers:
            self.model.add(layer)

        self.model.compile(
            loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"],
        )

    def train(
        self,
        model_dir,
        X,
        Y,
        X_validate,
        Y_validate,
        epochs=140,
        verbose=1,
        log_dir="logs",
    ):
        self.model.fit(
            self.datagen.flow(X, Y, batch_size=96),
            epochs=epochs,
            validation_data=self.validgen.flow(X_validate, Y_validate),
            shuffle=True,
            verbose=verbose,
            callbacks=[
                TensorBoard(
                    log_dir=os.path.join(model_dir, log_dir),
                    write_graph=True,
                    write_images=False,
                    histogram_freq=0,
                )
            ],
        )
        return self.model
