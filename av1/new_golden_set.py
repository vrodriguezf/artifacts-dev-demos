# Generates a training dataset and trains a few different (fake) models

import argparse
import json
import random

import wandb

parser = argparse.ArgumentParser()
parser.add_argument('datadir', type=str,
                    help='directory to read training data artifact from')

def upload_golden_set(datadir):
    run = wandb.init(job_type='create-golden-dataset', reinit=True)
    with run:
        run.log_artifact('dataset-test-main', paths=datadir, aliases=['golden'])

def main():
    args = parser.parse_args()
    upload_golden_set(args.datadir)

if __name__ == '__main__':
    main()
