# Generates a training dataset and trains a few different (fake) models

import argparse
import json
import random

import train

parser = argparse.ArgumentParser()
parser.add_argument('--datadir', type=str, required=True,
                    help='directory to read training data artifact from')


def main():
    args = parser.parse_args()

    for i in range(6):
        if random.random() < 0.5:
            train.train_model(args.datadir, 'stopsign')
        if random.random() < 0.6:
            train.train_model(args.datadir, 'pedestrian')
        if random.random() < 0.7:
            train.train_model(args.datadir, 'car')


if __name__ == '__main__':
    main()
