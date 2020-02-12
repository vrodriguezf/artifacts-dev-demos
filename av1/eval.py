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
        input_dataset = papi.artifact_version(dataset_path)
        run.log_input(input_dataset)
        input_model = papi.artifact_version(model_path)
        run.log_input(input_model)
        for i in range(5):
            for k in create_dataset.CLASSES:
                run.log({('%s-acc' % k): random.random() / (i + 1)})

def main():
    args = parser.parse_args()
    entity_name = iapi.settings('entity')
    project_name = iapi.settings('project')
    if entity_name is None:
        raise Exception('no entity')
    print('entity', entity_name)

    eval_model('test-pedestrian-golden',
      '%s/%s/model-pedestrian:latest' % (entity_name, project_name),
      '%s/%s/dataset-test-main:golden' % (entity_name, project_name))

    eval_model('test-stopsign-golden',
      '%s/%s/model-stopsign:latest' % (entity_name, project_name),
      '%s/%s/dataset-test-main:golden' % (entity_name, project_name))

    eval_model('test-car-golden',
      '%s/%s/model-car:latest' % (entity_name, project_name),
      '%s/%s/dataset-test-main:golden' % (entity_name, project_name))


if __name__ == '__main__':
    main()
