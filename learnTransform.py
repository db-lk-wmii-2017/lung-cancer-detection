import os
from tensorflow import keras
from keras.applications.vgg16 import VGG16
from keras_efficientnets import EfficientNetB0
from keras.optimizers import Adam
from utils import load_data, define_output_redirecter
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.layers import Dense, Flatten
import numpy as np

DATA_PATH = os.path.join("data", "model")


X, Y, X_test, Y_text = load_data(DATA_PATH, labels_as_categories=False, channels=3)

# X = preprocess_input(X)
# X_test = preprocess_input(X_test)


shape = (X.shape[1], X.shape[2], X.shape[3])
vgg16 = EfficientNetB0(include_top=False, input_shape=shape)
vgg16.trainable = False
flat1 = Flatten()(vgg16.output)
class1 = Dense(512, activation="relu")(flat1)
output = Dense(1, activation="softmax")(class1)

model = Model(inputs=vgg16.inputs, outputs=output)

model.compile(
    loss="binary_crossentropy", optimizer=Adam(), metrics=["accuracy"],
)

model.summary()

model.fit(
    X,
    Y,
    batch_size=96,
    epochs=10,
    validation_data=(X_test, Y_text),
    shuffle=True,
    verbose=1,
)
