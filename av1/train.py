# Generates a training dataset and trains a few different (fake) models

import argparse
import json
import random

import wandb

parser = argparse.ArgumentParser()
parser.add_argument('--datadir', type=str, required=True,
                    help='directory to read training data artifact from')
parser.add_argument('--model_type', type=str, default='stopsign',
                    help='what type of model are you training?')

def train_model(datadir, model_type):
    run = wandb.init(job_type='train-%s' % model_type, reinit=True)
    with run:
        run.config.update({'learning_rate': random.random()})
        run.use_artifact('dataset', path=datadir)

        for i in range(10):
            run.log({'loss': random.random() / (i + 1)})

        model_file = open('model.json', 'w')
        model_file.write('Model: %s\n%s\n' % (model_type, random.random()))
        model_file.close()

        run.log_artifact('model-%s' % model_type, paths='model.json', aliases='latest')

def main():
    args = parser.parse_args()
    train_model(args.datadir, args.model_type)

if __name__ == '__main__':
    main()
