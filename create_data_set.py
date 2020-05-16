#!/usr/bin/env python
# coding: utf-8

import os
import random
import argparse
from utils import for_each_sample, isCancerSample, save_data


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

def get_test(collection, size):
    test = []
    c = 0
    while(c < size):
        sample = None
        while True:
            sample = random.choices(collection, k=1)[0]
            if 'R' not in sample[-10:]:
                break
        test.append(sample)
        semi_id = sample[:-6]
        if 'R' in semi_id[-4:]:
            index = semi_id.rfind('R')
            semi_id = semi_id[:index]
            pass
        collection = list(filter(lambda x: semi_id not in x, collection))
        if (len(collection) == 0):
            raise Exception("Not enough data")
        c +=1
    return test, collection

def main():
    parser = argparse.ArgumentParser(description="Split dataset into train and test.")
    parser.add_argument(
        "-p",
        "--path",
        type=str,
        default=os.path.join("data", "preprocessed"),
        nargs="?",
        help="Relative path to proprocessed data",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=os.path.join("data", "model"),
        nargs="?",
        help="Output folder",
    )
    parser.add_argument(
        "-c",
        "--cancer",
        default=3004,
        type=int,
        nargs="?",
        help="Number of people with cancer",
    )
    parser.add_argument(
        "-nc",
        "--not-cancer",
        default=3004,
        type=int,
        nargs="?",
        help="Number of people without cancer",
    )
    parser.add_argument(
        "-tc",
        "--cancer-test",
        default=600,
        type=int,
        nargs="?",
        help="Number of people with cancer - test sample",
    )
    parser.add_argument(
        "-ntc",
        "--not-cancer-test",
        default=600,
        type=int,
        nargs="?",
        help="Number of people without cancer - test sample",
    )
    args = parser.parse_args()

    if not os.path.exists(args.path):
        print("Invalid path: {}".format(args.path))
        exit()

    cancer, not_cancer = get_all_samples(args.path)
    random.shuffle(cancer)    
    random.shuffle(not_cancer)

    print(
        "Loaded samples:\ncancer: {}\nnot cancer: {}".format(
            len(cancer), len(not_cancer)
        )
    )

    cancer_test, cancer = get_test(cancer, args.cancer_test)
    not_cancer_test, not_cancer = get_test(not_cancer, args.not_cancer_test)

    if len(cancer) < args.cancer or len(not_cancer) < args.not_cancer:
        raise Exception("Not enough data")

    cancer_train = select_random(cancer, args.cancer)
    not_cancer_train = select_random(not_cancer, args.not_cancer)

    train_data_set = cancer_train + not_cancer_train
    test_data_set = cancer_test + not_cancer_test

    random.shuffle(train_data_set)    
    random.shuffle(test_data_set)

    save_data(train_data_set, test_data_set, args.output)


if __name__ == "__main__":
    main()
