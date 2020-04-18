#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import argparse
from scipy import ndimage

parser = argparse.ArgumentParser(description='Show images.')
parser.add_argument('relativePath',
                    type=str,
                    default="data/preprocessed",
                    nargs='?',
                    help='Relative path to proprocessed data')
args = parser.parse_args()
PREPROCESSED_DATA_PATH = args.relativePath


def isCancerSample(sampleName):
    return int(sampleName.split('-')[1].split('.')[0]) == 1


rotated_num = 0
for patient in filter(lambda x: not x.startswith('.'),
                      os.listdir(PREPROCESSED_DATA_PATH)):
    samples = os.listdir(os.path.join(PREPROCESSED_DATA_PATH, patient))
    patientDir = os.path.join(PREPROCESSED_DATA_PATH, patient)
    for sample in os.listdir(patientDir):
        if isCancerSample(sample) and not 'R' in sample:
            print("Rotating: {} {}".format(patient, sample))
            rotated_num += 1
            sampleArray = np.load(
                os.path.join(PREPROCESSED_DATA_PATH, patient, sample))
            for deg in [90, 180, 270]:
                rotated = ndimage.rotate(sampleArray, deg, reshape=False)
                fileName = "R{}-".format(deg).join(sample.split('-'))
                np.save(os.path.join(patientDir, fileName), rotated)
print("Rotated: {}".format(rotated_num))
