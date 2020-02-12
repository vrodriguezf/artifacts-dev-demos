import argparse
import json
import random

import wandb

parser = argparse.ArgumentParser()

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
    'occloued': 0.4,
    'not': 0.6,
}

iapi = wandb.apis.InternalApi()
papi = wandb.Api()

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

def make_dataset_artifact(entity_name, project_name, artifact_name, n_examples):
    projects = papi.projects(entity_name)
    project = None
    for p in projects:
        if p.name == project_name:
            project = p
    if project is None:
        raise Exception('no project')

    artifact_id = None
    for a in project.artifacts():
        if a.name == artifact_name:
            artifact_id = a.id

    if artifact_id is None:
        artifact_id = iapi.create_artifact(entity_name, project_name, artifact_name)

    metadata = gen_metadata(n_examples)
    av = iapi.create_artifact_version(entity_name, project_name, None, artifact_id, metadata=json.dumps(metadata))
    return entity_name + '/' + project_name + '/' + artifact_name + ':' + av['name']


def train_model(dataset_path, job_type):
    run = wandb.init(job_type=job_type, reinit=True)
    with run:
        run.config.update({'learning_rate': random.random()})
        input_dataset = papi.artifact_version(dataset_path)
        run.log_input(input_dataset)
        for i in range(10):
            run.log({'loss': random.random() / (i + 1)})
        model_file = open('model.json', 'w')
        model_file.write('Model: %s' % job_type)
        model_file.close()
        run.log_artifact('model-%s' % job_type.lstrip('train-'), 'model.json')

def main():
    args = parser.parse_args()
    entity_name = iapi.settings('entity')
    project_name = iapi.settings('project')
    if entity_name is None:
        raise Exception('no entity')
    print('entity', entity_name)

    av_path = make_dataset_artifact(entity_name, project_name, 'dataset-train-main', 100121 + int(random.random() * 10000))
    print('AV', av_path)

    for i in range(6):
        if random.random() < 0.5:
            train_model(av_path, 'train-stopsign')
        if random.random() < 0.6:
            train_model(av_path, 'train-pedestrian')
        if random.random() < 0.7:
            train_model(av_path, 'train-car')


if __name__ == '__main__':
    main()
