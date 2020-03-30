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
import data_library_query

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

    chosen_cats = data_library_query.categories_filtered(
        args.supercategories, args.categories)
    chosen_cat_ids = [c['id'] for c in chosen_cats]

    labels = data_library_query.labels_of_types_and_categories(
        args.annotation_types, chosen_cat_ids)

    example_image_paths = set(l['image_path'] for l in labels)
    example_image_paths = set(random.sample(
        example_image_paths, int(args.select_fraction * len(example_image_paths))))
    if len(example_image_paths) == 0:
        print('Error, you must select at least 1 image')
        sys.exit(1)
    
    artifact = dataset.DatasetArtifact.from_library_query(
        example_image_paths, args.annotation_types)

    dataset_dir = './artifact'
    os.makedirs(dataset_dir, exist_ok=True)
    artifact.dump_files(dataset_dir)

    run.log_artifact(
        name='dataset/%s' % args.dataset_name,
        contents=dataset_dir,
        metadata={
            'annotation_types': args.annotation_types,
            'categories': [c['name'] for c in chosen_cats],
            'n_examples': len(example_image_paths)},
        aliases=[args.dataset_version] + ['latest'])


if __name__ == '__main__':
    main(sys.argv)