import os
import numpy as np
import sys

TRAIN_DATA_FILE_NAME = "train.txt"
TEST_DATA_FILE_NAME = "test.txt"


def define_output_redirecter():
    orig_stdout = sys.stdout
    f = None

    def redirect(file):
        try:
            if f != None:
                f.close()
        except:
            pass
        f = open(file, "w")
        sys.stdout = f

    def restore():
        sys.stdout = orig_stdout
        if f != None:
            f.close()

    return redirect, restore


def for_each_sample(path, fn):
    """
        fn(sample, patient, patientDir)
        - sample - sample file name
        - patient - patient id 
        - patientDir - path to patient folder
    """
    for patient in filter(lambda x: not x.startswith("."), os.listdir(path)):
        patientDir = os.path.join(path, patient)
        for sample in os.listdir(patientDir):
            fn(sample, patient, patientDir)


def isCancerSample(sampleName):
    return int(sampleName.split("-")[1].split(".")[0]) == 1


def save_data(train_data_set, test_data_set, path):
    save_to_csv_file(train_data_set, path, TRAIN_DATA_FILE_NAME)
    save_to_csv_file(test_data_set, path, TEST_DATA_FILE_NAME)


def save_to_csv_file(collection, path, file_name):
    if not os.path.exists(path):
        os.makedirs(path)
    full_path = os.path.join(path, file_name)
    print("Writing to: {}".format(full_path))
    with open(full_path, "w") as f:
        for line in collection:
            label = 1 if isCancerSample(line) else 0
            f.write("{} {}\n".format(line, label))


def load_data(path, labels_as_categories=True):
    X, Y = get_data_from_csv_file(
        os.path.join(path, TRAIN_DATA_FILE_NAME),
        labels_as_categories=labels_as_categories,
    )
    X_test, Y_text = get_data_from_csv_file(
        os.path.join(path, TEST_DATA_FILE_NAME),
        labels_as_categories=labels_as_categories,
    )
    return X, Y, X_test, Y_text


def get_data_from_csv_file(path, show_error=False, labels_as_categories=True):
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
                if labels_as_categories:
                    labels.append([0.0, 1.0] if int(label) else [1.0, 0.0])
                else:
                    labels.append(int(label))
            elif show_error:
                print("Shape error: {}".format(sample_path))
    return (
        np.asarray(data, dtype="f").reshape([-1, 50, 50, 1]),
        np.asarray(labels, dtype="f"),
    )
