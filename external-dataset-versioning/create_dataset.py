import argparse
import os
import random
import sys
import tempfile

from pycocotools.coco import COCO
import wandb

import dataset

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--supercategories', type=str, nargs='*',
                    help='coco supercategories to take examples from')
parser.add_argument('--categories', type=str, nargs='*',
                    help='coco categories to take examples from')
parser.add_argument('--select_fraction', type=float, default=1,
                    help='random fraction of examples to select')
parser.add_argument('--seed', type=int, default=0,
                    help='random seed')

parser.add_argument('--annotation_types', type=str, nargs='*', required=True,
                    choices=['bbox', 'segmentation', 'is_crowd'],
                    help='coco annotation types to include in dataset')


def main(argv):
    args = parser.parse_args()

    run = wandb.init(job_type='create_dataset')

    random.seed(args.seed)

    bucket_api = dataset.BucketApiLocal('bucket')
    ds_api = dataset.CocoDatasetAPI(bucket_api)

    coco_api = ds_api.get_coco_api()

    # Select image IDs for the requested categories
    cat_ids = coco_api.getCatIds(catNms=args.categories, supNms=args.supercategories)
    img_ids = coco_api.getImgIds(catIds=cat_ids)
    img_ids = random.sample(img_ids, int(args.select_fraction * len(img_ids)))
    categories = [c['name'] for c in coco_api.loadCats(cat_ids)]

    hashes = ds_api.get_content_hashes(img_ids)
    annotations = ds_api.get_annotations(img_ids, args.annotation_types)

    manifest = dataset.DatasetArtifactManifest()
    for img_id in img_ids:
        if img_id in annotations:
            manifest.set_example(img_id, hashes[img_id], annotations[img_id])

    manifest_path = 'dataset.json'
    manifest.dump(manifest_path)

    run.log_artifact('dataset',
        paths=manifest_path,
        metadata={
            'annotation_types': args.annotation_types,
            'categories': categories,
            'n_examples': len(img_ids)})



if __name__ == '__main__':
    main(sys.argv)