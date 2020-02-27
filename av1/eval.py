import argparse
import json
import random

import wandb

import create_dataset

parser = argparse.ArgumentParser()

iapi = wandb.apis.InternalApi()
papi = wandb.Api()


def eval_model(job_type, model_path, dataset_path):
    run = wandb.init(job_type=job_type, reinit=True)
    with run:
        run.use_artifact(dataset_path)
        run.use_artifact(model_path)
        for i in range(5):
            for k in create_dataset.CLASSES:
                run.log({('%s-acc' % k): random.random() / (i + 1)})

def main():
    args = parser.parse_args()

    eval_model('test-pedestrian-golden',
      'model-pedestrian:latest',
      'dataset-test-main:golden')

    eval_model('test-stopsign-golden',
      'model-stopsign:latest',
      'dataset-test-main:golden')

    eval_model('test-car-golden',
      'model-car:latest',
      'dataset-test-main:golden')


if __name__ == '__main__':
    main()
