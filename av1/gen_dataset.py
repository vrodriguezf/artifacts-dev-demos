import argparse
import json
import random
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument('dir', type=str,
                    help='directory to write metadata file to')
parser.add_argument('n_examples', type=int,
                    help='number of examples to generate metadata for')

WEATHER = {
    'clear': 0.4,
    'partly-cloudy': 0.2,
    'overcast': 0.3,
    'rainy': 0.2,
    'foggy': 0.1
}

SCENE = {
    'residential': 0.1,
    'highway': 0.2,
    'city-street': 0.3,
    'parking-lot': 0.05,
    'tunnel': 0.1
}

HOURS = {
    'dawn-dusk': 0.07,
    'daytime': 0.5,
    'night': 0.43
}


CLASSES = {
    'bus': 0.03,
    'light': 0.2,
    'sign': 0.3,
    'person': 0.4,
    'bike': 0.1,
    'truck': 0.2,
    'car': 0.9,
    'train': 0.001,
    'rider': 0.06,
}

OCCLUDED = {
    'occluded': 0.4,
    'not': 0.6,
}

def gen_metadata(n_examples):
    weather = {}
    for key, ratio in WEATHER.items():
        weather[key] = int((ratio + ratio * (random.random() - 0.5) * 0.2) * n_examples)
    scene = {}
    for key, ratio in SCENE.items():
        scene[key] = int((ratio + ratio * (random.random() - 0.5) * 0.2) * n_examples)
    hours = {}
    for key, ratio in HOURS.items():
        hours[key] = int((ratio + ratio * (random.random() - 0.5) * 0.2) * n_examples)
    class_counts = {}
    for key, ratio in CLASSES.items():
        class_counts[key] = int((ratio + ratio * (random.random() - 0.5) * 0.2) * n_examples)
    occluded = {}
    for key, ratio in OCCLUDED.items():
        occluded[key] = int((ratio + ratio * (random.random() - 0.5) * 0.2) * n_examples)
    return {
        'summary': {
            'n_examples': n_examples,
            'weather': weather,
            'scene': scene,
            'hours': hours,
            'class_counts': class_counts,
            'occluded': occluded
        }
    }

def main():
    args = parser.parse_args()
    try:
        os.makedirs(args.dir)
    except FileExistsError:
        pass
    f = open(os.path.join(args.dir, 'artifact-metadata.json'), 'w')
    json.dump(gen_metadata(args.n_examples), f)
    f.write('\n')
    

if __name__ == '__main__':
    main()