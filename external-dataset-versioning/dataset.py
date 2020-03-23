import json
import os
import hashlib
import shutil
import tempfile
from pycocotools import coco

import boto3


class BucketApiS3(object):
    def __init(self, bucket_name):
        self._bucket_name = bucket_name
        self._s3 = boto3.client('s3')
    
    def download_file(self, key, local_path):
        self._s3.download_file(self._bucket_name, key, local_path)

    def get_hash(self, key):
        head_obj = self._s3.head_object(Bucket=self._bucket_name, Key=key)
        return head_obj['ETag']


class BucketApiLocal(object):
    def __init__(self, local_dir):
        self._local_dir = local_dir
    
    def download_file(self, key, local_path):
        shutil.copyfile(os.path.join(self._local_dir, key), local_path)

    def get_hash(self, key):
        hash_md5 = hashlib.md5()
        with open(os.path.join(self._local_dir, key), "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()


class CocoDatasetAPI(object):
    ANNOTATIONS_PATH = 'annotations/instances_val2017.json'

    def __init__(self, bucket_api):
        self._bucket_api = bucket_api

    def download_examples(self, dir, ids):
        os.makedirs(dir, exist_ok=True)
        for id in ids:
            path = self.get_example_path(id)
            _, ext = os.path.splitext(path)
            self._bucket_api.download_file(path, '%s.%s' % (id, ext))

    def get_example_path(self, id):
        return 'val2017/%012u.jpg' % id

    def get_content_hashes(self, ids):
        hashes = {}
        for id in ids:
            path = self.get_example_path(id)
            hashes[id] = self._bucket_api.get_hash(path)
        return hashes

    def get_coco_api(self):
        with tempfile.TemporaryDirectory() as save_dir:
            anno_path = os.path.join(save_dir, 'annotations.json')
            self._bucket_api.download_file(self.ANNOTATIONS_PATH, anno_path)
            return coco.COCO(anno_path)

    def get_annotations(self, ids, types=[]):
        ann_keys = ['category_id']
        ann_keys.extend(types)
        coco_api = self.get_coco_api()
        ann_ids = coco_api.getAnnIds(imgIds=ids)
        anns = coco_api.loadAnns(ann_ids)

        ann_result = {}
        for ann in anns:
            dataset_ann = {}
            for k, v in ann.items():
                if k in ann_keys:
                    dataset_ann[k] = v
            if len(dataset_ann) == len(ann_keys):
                image_id = ann['image_id']
                if image_id not in ann_result:
                    ann_result[image_id] = []
                ann_result[image_id].append(dataset_ann)
        return ann_result


class DatasetArtifactManifest(object):
    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls(json.load(f))

    def __init__(self, examples={}):
        self.examples = examples

    def set_example(self, example_id, hash, annotations):
        self.examples[example_id] = {'hash': hash, 'annotations': annotations}

    def dump(self, path):
        with open(path, 'w') as f:
            json.dump(self.examples, f, indent=2, sort_keys=True)

# TODO:
# - make script to consume artifacts (training script)
# - make check if updated function
# - make some examples of updating labels
# - make examples of adding images
# - write documentation / guide
