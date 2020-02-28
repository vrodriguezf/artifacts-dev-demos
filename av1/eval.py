import argparse
import json
import random

import wandb

import gen_dataset

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default='model-stopsign:latest',
                    help='what type of model are you training?')


def eval_model(job_type, model_path, dataset_path):
    run = wandb.init(job_type=job_type, reinit=True)
    with run:
        run.use_artifact(dataset_path)
        run.use_artifact(model_path)
        for i in range(5):
            for k in gen_dataset.CLASSES:
                run.log({('%s-acc' % k): random.random() / (i + 1)})

def main():
    args = parser.parse_args()
    eval_model('eval-golden', args.model, 'dataset-test-main:golden')


if __name__ == '__main__':
    main()
