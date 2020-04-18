#!/usr/bin/env python
# coding: utf-8

import os
import random
import argparse
from utils import for_each_sample, isCancerSample


def get_all_samples(path):
    cancer = []
    not_cancer = []

    def handle_sample(sample, patient, patientDir):
        global isCancerSample
        path = os.path.join(patientDir, sample)
        if isCancerSample(sample):
            cancer.append(path)
        else:
            not_cancer.append(path)

    for_each_sample(path, handle_sample)
    return cancer, not_cancer


def select_random(collection, size):
    return random.choices(collection, k=size)


def split_into_train_test(collection, train_percentage):
    size = int(len(collection) * train_percentage)
    return collection[:-size], collection[-size:]


def save(collection, path, file_name):
    if not os.path.exists(path):
        os.makedirs(path)
    full_path = os.path.join(path, file_name)
    print("Writing to: {}".format(full_path))
    with open(full_path, 'w') as f:
        for line in collection:
            label = 1 if isCancerSample(line) else 0
            f.write("{} {}\n".format(line, label))


def main():
    parser = argparse.ArgumentParser(
        description='Split dataset into train and test.')
    parser.add_argument('-p',
                        '--path',
                        type=str,
                        default="data/preprocessed",
                        nargs='?',
                        help='Relative path to proprocessed data')
    parser.add_argument('-o',
                        '--output',
                        type=str,
                        default="data/model",
                        nargs='?',
                        help='Output folder')
    parser.add_argument('-c',
                        '--cancer',
                        default=4,
                        type=int,
                        nargs='?',
                        help='Number of people with cancer')
    parser.add_argument('-nc',
                        '--not-cancer',
                        default=4,
                        type=int,
                        nargs='?',
                        help='Number of people without cancer')
    parser.add_argument('-tc',
                        '--cancer-train',
                        default=0.1,
                        type=float,
                        nargs='?',
                        help='Percentage intended for train data - cancer')
    parser.add_argument('-ntc',
                        '--not-cancer-train',
                        default=0.1,
                        type=float,
                        nargs='?',
                        help='Percentage intended for train data - not cancer')
    args = parser.parse_args()

    if not os.path.exists(args.path):
        print("Invalid path: {}".format(args.path))
        exit()

    cancer, not_cancer = get_all_samples(args.path)

    print("Loaded samples:\ncancer: {}\nnot cancer: {}".format(
        len(cancer), len(not_cancer)))
    if len(cancer) < args.cancer or len(not_cancer) < args.not_cancer:
        print("Not enough data")
        exit()

    cancer_data_set = select_random(cancer, args.cancer)
    not_cancer_data_set = select_random(not_cancer, args.not_cancer)

    cancer_train, cancer_test = split_into_train_test(cancer_data_set,
                                                      args.cancer_train)
    not_cancer_train, not_cancer_test = split_into_train_test(
        not_cancer_data_set, args.not_cancer_train)

    train_data_set = cancer_train + not_cancer_train
    test_data_set = cancer_test + not_cancer_test

    random.shuffle(train_data_set)

    save(train_data_set, args.output, "train.txt")
    save(test_data_set, args.output, "test.txt")


if __name__ == "__main__":
    main()
