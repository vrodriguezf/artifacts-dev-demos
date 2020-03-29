import argparse
import collections
import os
import random
import sys
import tempfile

import wandb

import dataset
import bucket_api
import data_library

parser = argparse.ArgumentParser(description='Train a new model, based on a dataset artifact.')
parser.add_argument('--dataset', type=str, required=True, help='')
parser.add_argument('--model_type', required=True,
    type=str, choices=['bbox', 'segmentation'], help='')


def main(argv):
    args = parser.parse_args()
    run = wandb.init(job_type='train-%s' % args.model_type)
    run.config.update(args)
    ds = run.use_artifact(args.dataset)
    if args.model_type not in ds.metadata['annotation_types']:
        print('Dataset %s has annotations %s, can\'t train model type: %s' % (
            args.dataset, ds.metadata['annotation_types'], args.model_type))
        sys.exit(1)
    datadir = ds.download()

    for i in range(10):
        run.log({'loss': random.random() / (i + 1)})

    model_file = open('model.json', 'w')
    model_file.write(
        'This is a placeholder. In a real job, you\'d save model weights here\n%s\n' % random.random())
    model_file.close()

    run.log_artifact('model-%s' % args.model_type, paths='model.json', aliases='latest')

if __name__ == '__main__':
    main(sys.argv)