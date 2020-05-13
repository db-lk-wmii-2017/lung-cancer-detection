#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import argparse
from tqdm import trange, tqdm
from scans import ScansReader, PatientInfoProvider

EXT = "npy"


def extract_features(data_path, csv_path, output_path, skip_preprocessed):
    skiped = 0
    errors = 0
    preprocessed = 0
    nodes = 0

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    reader = ScansReader(data_path, PatientInfoProvider(csv_path))
    for index, scan in enumerate(tqdm(reader.load())):
        suspicious_places = scan.get_suspicious_places()

        if scan.is_empty():
            print("Cound't find csv data for {}".format(file_name))
            errors += 1
            continue
        pass

        path = os.path.join(output_path, scan.get_name())
        if not os.path.exists(path):
            os.makedirs(path)
        elif skip_preprocessed:
            skiped += 1
            continue

        preprocessed += 1
        for i, (image, clazz) in enumerate(
            tqdm(scan.extract_images(), total=scan.size())
        ):
            nodes += 1
            np.save(os.path.join(path, "{}-{}.{}".format(i, clazz, EXT)), image)

    return preprocessed, nodes, skiped, errors


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def main():
    parser = argparse.ArgumentParser(description="Preprocess scans.")
    parser.add_argument(
        "-d",
        "--dataPath",
        type=str,
        default=os.path.join("data", "sample"),
        nargs="?",
        help="Relative path to raw data",
    )
    parser.add_argument(
        "-c",
        "--csvPath",
        type=str,
        default=os.path.join("data", "sample", "node.csv"),
        nargs="?",
        help="Relative path to csv file",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=os.path.join("data", "preprocessed"),
        nargs="?",
        help="Relative path to output folder",
    )
    parser.add_argument(
        "-s",
        "--skipPreprocessed",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Skip preprocessed.",
    )
    args = parser.parse_args()

    preprocessed, nodes, skiped, errors = extract_features(
        args.dataPath, args.csvPath, args.output, args.skipPreprocessed
    )
    print(
        "\n\nPreprocessed: {}\n  > Nodes {}\nSkiped: {}\nError: {}".format(
            preprocessed, nodes, skiped, errors
        )
    )


if __name__ == "__main__":
    main()
