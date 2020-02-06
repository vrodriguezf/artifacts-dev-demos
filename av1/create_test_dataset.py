import argparse
import json
import random

import wandb

import create_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--artifact_name', type=str, required=True)
parser.add_argument('--n_examples', type=int, required=True)

iapi = wandb.apis.InternalApi()
papi = wandb.Api()


def main():
    args = parser.parse_args()
    entity_name = iapi.settings('entity')
    project_name = iapi.settings('project')
    if entity_name is None:
        raise Exception('no entity')

    create_dataset.make_dataset_artifact(entity_name, project_name, args.artifact_name, args.n_examples)


if __name__ == '__main__':
    main()