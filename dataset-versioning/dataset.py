import json
import os
import random
import string

import wandb

import bucket_api
import data_library_query


IMAGES_FNAME = 'images.json'
LABELS_FNAME = 'labels.json'

def random_string(n):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(n))


class Dataset(object):
    """Methods for reading and writing the files stored in our dataset artifacts.

    For each example we store a path to the image in our data library (and its checksum)
    and the specific labels needed for this dataset.
    """
    @classmethod
    def from_artifact(cls, artifact):
        artifact_dir = artifact.download()
        examples = json.load(open(os.path.join(artifact_dir, IMAGES_FNAME)))
        labels = json.load(open(os.path.join(artifact_dir, LABELS_FNAME)))
        return cls(examples, labels, artifact=artifact)

    @classmethod
    def from_library_query(cls, example_image_paths, label_types):
        bucketapi = bucket_api.get_bucket_api()
        examples = [
            [path, bucketapi.get_hash(path)] for path in example_image_paths]
        labels = data_library_query.labels_for_images(
            example_image_paths, label_types)
        return cls(examples, labels)
    
    def __init__(self, examples, labels, artifact=None):
        self._artifact = artifact
        self.examples = sorted(examples)
        self.labels = sorted(labels, key=lambda l: (l['id'], 'bbox' in l))

    @property
    def example_image_paths(self):
        return set([e[0] for e in self.examples])

    def download(self):
        datadir = self.artifact.external_data_dir
        bucketapi = bucket_api.get_bucket_api()
        for example in self.examples:
            image_path, image_hash = example
            bucketapi.download_file(image_path, os.path.join(datadir, image_path), hash=image_hash)
        return datadir

    @property
    def artifact(self):
        if self._artifact is None:
            self._artifact = self.to_artifact()
        return self._artifact

    def to_artifact(self):
        # TODO: add more metadata (class dist)
        annotation_types = []
        if 'bbox' in self.labels[0]:
            annotation_types.append('bbox')
        if 'segmentation' in self.labels[0]:
            annotation_types.append('segmentation')

        artifact = wandb.WriteableArtifact(
            type='dataset',
            metadata= {
                'n_examples': len(self.examples),
                'annotation_types': annotation_types})

        with open(os.path.join(artifact.artifact_dir, IMAGES_FNAME), 'w') as f:
            json.dump(self.examples, f, indent=2, sort_keys=True)
        with open(os.path.join(artifact.artifact_dir, LABELS_FNAME), 'w') as f:
            json.dump(self.labels, f, indent=2, sort_keys=True)

        return artifact