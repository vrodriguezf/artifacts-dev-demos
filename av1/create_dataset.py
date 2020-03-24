import argparse
import binascii
import json
import random
import os
import sys

import wandb


parser = argparse.ArgumentParser()
parser.add_argument('--datadir', type=str, required=True,
                    help='directory to write metadata file to')
parser.add_argument('--n_examples', type=int, required=True,
                    help='number of examples to generate metadata for')
parser.add_argument('--golden', action='store_true',
                    help='apply the "golden" alias to this dataset')

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
    run = wandb.init(job_type='create_dataset')

    args = parser.parse_args()

    # We store the path to the actual data inside the artifact. For this example,
    # this is not a real url.
    external_id = binascii.b2a_hex(os.urandom(15))
    artifact_contents = {'url': 's3://example.com/path/to/data/%s' %
        external_id.decode('utf-8')}

    # Write the artifact-contents file.
    os.makedirs(args.datadir, exist_ok=True)
    path = os.path.join(args.datadir, 'artifact-contents.json')
    with open(path, 'w') as artifact_contents_file:
        json.dump(artifact_contents, artifact_contents_file, indent=2)
        artifact_contents_file.write('\n')

    # save the artifact, along with useful metadata about it's contents
    metadata = gen_metadata(args.n_examples)
    aliases = None
    if args.golden:
        aliases = ['golden']
    run.log_artifact('dataset', paths=args.datadir, metadata=metadata, aliases=aliases)
    

if __name__ == '__main__':
    main()