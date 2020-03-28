import argparse
import collections
import os
import random
import sys
import tempfile

import wandb

import dataset
import bucket_api
import data_library

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--dataset_name', type=str, required=True,
                    help='name for this dataset')
parser.add_argument('--dataset_version', type=str, required=True,
                    help='version label for this dataset')
parser.add_argument('--supercategories', type=str, nargs='*', default=[],
                    help='coco supercategories to take examples from')
parser.add_argument('--categories', type=str, nargs='*', default=[],
                    help='coco categories to take examples from')
parser.add_argument('--select_fraction', type=float, default=1,
                    help='random fraction of examples to select')
parser.add_argument('--seed', type=int, default=0,
                    help='random seed')
parser.add_argument('--annotation_types', type=str, nargs='*', required=True,
                    choices=['bbox', 'segmentation'],
                    help='coco annotation types to include in dataset')


def main(argv):
    args = parser.parse_args()

    run = wandb.init(job_type='create_dataset')

    random.seed(args.seed)

    bucketapi = bucket_api.get_bucket_api()

    categories = data_library.get_categories()

    chosen_cats = [c for c in categories
        if c['supercategory'] in args.supercategories or
           c['name'] in args.categories]
    chosen_cat_ids = [c['id'] for c in chosen_cats]

    example_labels = collections.defaultdict(list)
    labels = []

    if 'bbox' in args.annotation_types:
        box_labels = data_library.get_box_labels()
        for bl in box_labels.values():
            if bl['category_id'] in chosen_cat_ids:
                labels.append(bl)
    if 'segmentation' in args.annotation_types:
        seg_labels = data_library.get_seg_labels()
        for sl in seg_labels.values():
            if sl['category_id'] in chosen_cat_ids:
                labels.append(sl)

    example_image_paths = set(l['image_path'] for l in labels)
    example_image_paths = set(random.sample(
        example_image_paths, int(args.select_fraction * len(example_image_paths))))
    labels = [l for l in labels if l['image_path'] in example_image_paths]
    
    examples = [(path, bucketapi.get_hash(path)) for path in example_image_paths]
    artifact = dataset.DatasetArtifact(examples, labels)
    # print('len(examples)', examples)

    dataset_dir = './artifact'
    os.makedirs(dataset_dir, exist_ok=True)
    artifact.dump_files(dataset_dir)

    run.log_artifact(args.dataset_name,
        paths=dataset_dir,
        metadata={
            'annotation_types': args.annotation_types,
            'categories': [c['name'] for c in chosen_cats],
            'n_examples': len(examples)},
        aliases=[args.dataset_version] + ['latest'])



if __name__ == '__main__':
    main(sys.argv)