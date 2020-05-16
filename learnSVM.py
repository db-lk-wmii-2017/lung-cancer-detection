import os
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from utils import load_data, define_output_redirecter
import numpy as np
import pickle

DATA_PATH = os.path.join("data", "model")
MODEL_OUTPUT = os.path.join("data", "SVMModel")

if not os.path.exists(MODEL_OUTPUT):
    os.makedirs(MODEL_OUTPUT)

redirect_output, restore_output = define_output_redirecter()

X, Y, X_test, Y_text = load_data(DATA_PATH, labels_as_categories=False,)
X = X / 255.0
X_test = X_test / 255.0

X = np.reshape(X, [-1, 50 * 50])
X_test = np.reshape(X_test, [-1, 50 * 50])

scaler = StandardScaler()
scaler.fit(X)

param_grid = {
    "C": [0.1, 1, 10, 100, 1000],
    "gamma": [1, 0.1, 0.01, 0.001, 0.0001],
    "kernel": ["rbf", "poly", "linear"],
    "degree": [3, 5, 8, 10, 12, 30],
}

grid = GridSearchCV(SVC(), param_grid, verbose=2)

grid.fit(scaler.transform(X), Y)

redirect_output(os.path.join(MODEL_OUTPUT, "svm" + ".log"))

print(grid.best_params_)
print()
print(grid.best_estimator_)
print()
predictions = grid.predict(scaler.transform(X_test))
print(classification_report(Y_text, predictions))

pickle.dump(grid.best_estimator_, open(os.path.join(MODEL_OUTPUT, "svm.pkl"), "wb"))

restore_output()
