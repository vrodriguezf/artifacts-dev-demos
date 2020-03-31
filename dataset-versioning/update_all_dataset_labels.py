"""Create new versions of datasets that need updates."""

import argparse
import collections
import json
import os
import random
import sys
import tempfile

import wandb

import dataset
import data_library

parser = argparse.ArgumentParser(description='Create new versions of datasets if new labels are available.')


def main(argv):
    args = parser.parse_args()

    # TODO: Make it easier to contruct the api
    api_settings = wandb.InternalApi().settings()
    api = wandb.Api(api_settings)

    # Initialize a W&B run
    run = wandb.init(job_type='update_dataset')
    run.config.update(args)

    # iterate through all the datasets we have.
    datasets = api.artifact_types('%s/%s' % (api_settings['entity'], api_settings['project']))
    # TODO: switch to one upload job per artifact, and only if we're upgrading, which
    #     means we need to use the public API before the run API
    for d in datasets:
        # TODO: query for only dataset artifacts instead of filtering here.
        if d.name.startswith('model'):
            continue
        print('Checking latest for dataset: %s' % d)

        # fetch the latest version of each dataset artifact and download it's contents
        ds_artifact = run.use_artifact('%s:latest' % d.name)
        datadir = ds_artifact.download()
        ds_contents = dataset.DatasetArtifactContents.from_dir(datadir)

        # construct dataset artifact contents using the example in the loaded dataset,
        # but with the most recent labels from the library.
        library_ds_contents = dataset.DatasetArtifactContents.from_library_query(
            ds_contents.example_image_paths,
            ds_artifact.metadata['annotation_types'])

        # if the contents aren't equal, then create a new version based on
        # library_ds_contents
        if ds_contents != library_ds_contents:
            print('  updated, create new dataset version')
            dataset_dir = './artifact/%s' % ds_artifact.artifact_type_name
            os.makedirs(dataset_dir, exist_ok=True)
            library_ds_contents.dump_files(dataset_dir)
            run.log_artifact(
                name=ds_artifact.artifact_type_name,
                contents=dataset_dir,
                metadata=ds_artifact.metadata,
                # TODO: bump version number instead of hard-coding to v2
                aliases=['v2', 'latest'])


if __name__ == '__main__':
    main(sys.argv)