import json
import os


class DatasetArtifact(object):
    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls(json.load(f))

    def __init__(self, examples, labels):
        self.examples = examples
        self.labels = labels

    def dump_files(self, dirpath):
        with open(os.path.join(dirpath, 'images.json'), 'w') as f:
            json.dump(sorted(self.examples),
            f, indent=2, sort_keys=True)
        with open(os.path.join(dirpath, 'labels.json'), 'w') as f:
            json.dump(sorted(self.labels,
                key=lambda l: (l['id'], 'bbox' in l)),
                f, indent=2, sort_keys=True)