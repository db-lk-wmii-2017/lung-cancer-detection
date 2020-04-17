#!/usr/bin/env python3

import sys
import os
import numpy as np
import argparse
import SimpleITK as sitk
import functools
from PIL import Image


def get_chunks(l, n):
    n = max(1, n)
    return (l[i:i + n] for i in range(0, len(l), n))


def get_concat_h(im1, im2):
    dst = Image.new('L', (im1.width + im2.width, max(im1.height, im2.height)))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


def get_concat_v(im1, im2):
    dst = Image.new('L', (max(im1.width, im2.width), im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst


def showNumpyArray(filePath, width, height):
    array = np.load(filePath)
    Image.fromarray(array).resize((width, height)).show()


def showScan(filePath, width, height):
    ct_scan = sitk.GetArrayFromImage(sitk.ReadImage(filePath))
    images = list(
        map(lambda x: Image.fromarray(x).resize((width, height)), ct_scan))
    chunks = get_chunks(images, 20)
    result = Image.new('L', (0, 0))
    for chunk in chunks:
        row = Image.new('L', (0, 0))
        for image in chunk:
            row = get_concat_h(row, image)
        result = get_concat_v(result, row)
    result.show(title=filePath)


parser = argparse.ArgumentParser(description='Show images.')
parser.add_argument('relativePath', type=str, help='Relative file/folder path')
parser.add_argument('-w',
                    '--width',
                    default=50,
                    type=int,
                    help='Width in inches.')
parser.add_argument('-l',
                    '--height',
                    default=50,
                    type=int,
                    help='Height in inches.')

args = parser.parse_args()

path = os.path.join(os.getcwd(), args.relativePath)

if not os.path.exists(path):
    print("File don't exists: {}".format(path))
    exit()

if path.endswith('npy'):
    showNumpyArray(path, args.width, args.height)
elif path.endswith('mhd'):
    showScan(path, args.width, args.height)
else:
    print("Unknown file extension: {}".format(args.relativePath))
    exit()
