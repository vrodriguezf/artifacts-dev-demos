import json
import os
import random
import string

import bucket_api
import data_library_query

IMAGES_FNAME = 'images.json'
LABELS_FNAME = 'labels.json'

def random_string(n):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(n))


class DatasetArtifactContents(object):
    """Methods for reading and writing the files stored in our dataset artifacts.

    For each example we store a path to the image in our data library (and its checksum)
    and the specific labels needed for this dataset.
    """
    @classmethod
    def from_dir(cls, dirpath):
        examples = json.load(open(os.path.join(dirpath, IMAGES_FNAME)))
        labels = json.load(open(os.path.join(dirpath, LABELS_FNAME)))
        return cls(examples, labels)

    @classmethod
    def from_library_query(cls, example_image_paths, label_types):
        bucketapi = bucket_api.get_bucket_api()
        examples = [
            [path, bucketapi.get_hash(path)] for path in example_image_paths]
        labels = data_library_query.labels_for_images(
            example_image_paths, label_types)
        return cls(examples, labels)
    
    def __init__(self, examples, labels):
        self.examples = sorted(examples)
        self.labels = sorted(labels, key=lambda l: (l['id'], 'bbox' in l))

    def __eq__(self, other): 
        return self.examples == other.examples and self.labels == other.labels

    @property
    def example_image_paths(self):
        return set([e[0] for e in self.examples])

    def download(self, rootdir='./datasets'):
        # TODO: use dataset name as dirname, don't redownload if checksums pass
        datadir = os.path.join(rootdir, random_string(8))
        bucketapi = bucket_api.get_bucket_api()
        for example in self.examples:
            image_path = example[0]
            bucketapi.download_file(image_path, os.path.join(datadir, image_path))
            # TODO: confirm checksum
        return datadir

    def dump_files(self, dirpath):
        with open(os.path.join(dirpath, IMAGES_FNAME), 'w') as f:
            json.dump(self.examples, f, indent=2, sort_keys=True)
        with open(os.path.join(dirpath, LABELS_FNAME), 'w') as f:
            json.dump(self.labels, f, indent=2, sort_keys=True)