#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import argparse
from preprocess import rotate
from utils import for_each_sample, isCancerSample

# Globals
rotated_num = 0


def handle_sample(sample, patient, patientDir):
    global rotated_num
    if isCancerSample(sample) and not "R" in sample:
        print("Rotating: {} {}".format(patient, sample))
        rotated_num += 1
        sampleArray = np.load(os.path.join(patientDir, sample))
        for deg in [90, 180, 270]:
            rotated = rotate(sampleArray, deg)
            fileName = "R{}-".format(deg).join(sample.split("-"))
            np.save(os.path.join(patientDir, fileName), rotated)


def main():
    parser = argparse.ArgumentParser(description="Rotate images.")
    parser.add_argument(
        "relativePath",
        type=str,
        default=os.path.join("data", "preprocessed"),
        nargs="?",
        help="Relative path to proprocessed data",
    )
    args = parser.parse_args()

    for_each_sample(args.relativePath, handle_sample)
    print("Rotated: {}".format(rotated_num))


if __name__ == "__main__":
    main()
