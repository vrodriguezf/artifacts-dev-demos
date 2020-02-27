import argparse
import json
import random

import wandb

import create_dataset

parser = argparse.ArgumentParser()

iapi = wandb.apis.InternalApi()
papi = wandb.Api()


def main():
    run = wandb.init(job_type='create-test-dataset')
    run.log_artifact('dataset-test-main',
        metadata=create_dataset.gen_metadata(1202 + random.random() * 100),
        aliases=['golden'])


if __name__ == '__main__':
    main()