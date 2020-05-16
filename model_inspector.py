#!/usr/bin/env python
# coding: utf-8

import os
from tensorflow import keras

model_path = os.path.join("data", "model18")

model_names = filter(lambda x: x.endswith(".tf1"), os.listdir(model_path))
for model_name in model_names:
    print(model_name)
    print(os.path.join(model_path, model_name))
    model = keras.models.load_model(os.path.join(model_path, model_name))
    model.summary()
