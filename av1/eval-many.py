import argparse
import eval
import json
import random


def main():
    eval.eval_model('eval-golden',
       'model-pedestrian:latest',
       'dataset:golden')

    eval.eval_model('eval-golden',
       'model-stopsign:latest',
       'dataset:golden')

    eval.eval_model('eval-golden',
       'model-car:latest',
       'dataset:golden')


if __name__ == '__main__':
    main()
