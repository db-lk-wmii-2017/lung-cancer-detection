from tensorflow import keras
from keras import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import BatchNormalization
from keras.layers import Conv2D, Flatten, MaxPooling2D, Dropout, BatchNormalization, Dense, InputLayer
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
class CNN(object):
    def __init__(self, network=None):
        self.model = Sequential()
        self.datagen = None
        self.validgen = None

    def init_input_layer(self, X):
        def preprocessing_function(image):
            if bool(random.getrandbits(1)):
                sigma = random.uniform(0., 3.0)
                image = scipy.ndimage.filters.gaussian_filter(image, sigma)
            return image

        self.datagen = ImageDataGenerator(
            featurewise_center=True,
            horizontal_flip=True,
            rotation_range=25,
            featurewise_std_normalization=True,
            preprocessing_function=preprocessing_function, 
        )
		
        self.validgen = ImageDataGenerator(
            featurewise_center=True,
            featurewise_std_normalization=True,
        )

        self.datagen.fit(X)
        self.validgen.fit(X)
        shape = (X.shape[1], X.shape[2], X.shape[3])
        self.model.add(InputLayer(input_shape=shape))
        pass

    def define_network(self, X):
        self.init_input_layer(X)
        self.model.add(
            Conv2D(
                32,
                kernel_size=(5, 5),
                activation=activations.relu,
                name="conv1",
                kernel_regularizer=regularizers.l2(), 
                bias_regularizer=regularizers.l2(),
                padding="same",
            )
        )
        self.model.add(MaxPooling2D(pool_size=(2, 2), name="mp1", padding="same"))
        self.model.add(
            Conv2D(
                64,
                kernel_size=(5, 5),
                activation=activations.relu,
                name="conv2",
                kernel_regularizer=regularizers.l2(), 
                bias_regularizer=regularizers.l2(),
                padding="same",
            )
        )
        self.model.add(
            Conv2D(
                64,
                kernel_size=(3, 3),
                activation=activations.relu,
                name="conv3",
                kernel_regularizer=regularizers.l2(), 
                bias_regularizer=regularizers.l2(),
                padding="same",
            )
        )
        #self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(2, 2), name="mp2", padding="same"))
        self.model.add(Flatten())
        self.model.add(Dense(
            512,
            activation=activations.relu,
            name="dense1",
        ))
        self.model.add(Dropout(
            0.5,
            name="dp1"
        ))
        self.model.add(Dense(
            2,
            activation=activations.softmax,
            name="dense2",
        ))

        self.model.compile(
            loss="categorical_crossentropy",
            optimizer = Adam(learning_rate=0.0001),
            metrics=['accuracy'])
    

    def train(self, model_dir, X, Y, X_validate, Y_validate):
        logdir = os.path.join(model_dir, "logs") 
        tensorboard_callback = TensorBoard(log_dir=logdir, write_graph=True, write_images=True, histogram_freq=0)
        self.model.fit(
            self.datagen.flow(X, Y, batch_size=96),
            epochs=140,
            #validation_data=(X_validate, Y_validate),
			validation_data=self.validgen.flow(X_validate, Y_validate),
            shuffle=True,
            verbose=0,
            callbacks=[tensorboard_callback],
			#learning_rate = 0.001
        )
        return self.model
