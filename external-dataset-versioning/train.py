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

    ds_artifact = run.use_artifact('dataset/%s' % args.dataset)
    if args.model_type not in ds_artifact.metadata['annotation_types']:
        print('Dataset %s has annotations %s, can\'t train model type: %s' % (
            args.dataset, ds_artifact.metadata['annotation_types'], args.model_type))
        sys.exit(1)

    artifact_dir = ds_artifact.download()
    ds = dataset.DatasetArtifact.from_dir(artifact_dir)
    dataset_dir = ds.download()

    X, y = [], []
    for label in ds.labels:
        X.append(os.path.join(dataset_dir, label['image_path']))
        y.append(label['bbox'])
    print ('Xlen, ylen', len(X), len(y))

    for i in range(10):
        run.log({'loss': random.random() / (i + 1)})

    model_file = open('model.json', 'w')
    model_file.write(
        'This is a placeholder. In a real job, you\'d save model weights here\n%s\n' % random.random())
    model_file.close()

    run.log_artifact(
        name='model/%s' % args.model_type,
        contents='model.json',
        aliases='latest')

if __name__ == '__main__':
    main(sys.argv)