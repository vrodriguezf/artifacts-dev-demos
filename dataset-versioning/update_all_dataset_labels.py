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

    # TODO: ew
    api_settings = wandb.InternalApi().settings()
    api = wandb.Api(api_settings)

    print(api_settings)
    datasets = api.artifact_types('%s/%s' % (api_settings['entity'], api_settings['project']))
    # TODO: switch to one upload job per artifact, and only if we're upgrading, which
    #     means we need to use the public API before the run API
    for d in datasets:
        # TODO: ew. We need a higher level type
        if d.name.startswith('model'):
            continue
        print('Checking latest for dataset: %s' % d)

        run = wandb.init(job_type='update_dataset')
        run.config.update(args)
        ds_artifact = run.use_artifact('%s:latest' % d.name)
        datadir = ds_artifact.download()

        ds = dataset.DatasetArtifact.from_dir(datadir)

        library_ds = dataset.DatasetArtifact.from_library_query(
            ds.example_image_paths, ds_artifact.metadata['annotation_types'])

        if ds != library_ds:
            print('  updated, create new dataset version')
            dataset_dir = './artifact/%s' % ds_artifact.artifact_type_name
            os.makedirs(dataset_dir, exist_ok=True)
            library_ds.dump_files(dataset_dir)
            run.log_artifact(
                name=ds_artifact.artifact_type_name,
                contents=dataset_dir,
                metadata=ds_artifact.metadata,
                # TODO: bump version number
                aliases=['v2', 'latest'])


if __name__ == '__main__':
    main(sys.argv)