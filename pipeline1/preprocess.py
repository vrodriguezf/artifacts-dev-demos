import glob
import os
import sys
import wandb

api = wandb.Api()

run = wandb.init(job_type='preprocess')

# most recent raw dataset
dataset = api.artifact_version('shawn/artifacts6/kv8rthsdd')

print('Dataset', dataset)

dataset_dir = dataset.download()

run.log_input(dataset)

all_data = []
for fpath in glob.glob(os.path.join(dataset_dir, 'dataset', '*')):
    f = open(fpath)
    all_data.append(f.read())

out = open('alldata.txt', 'w')
for d in all_data:
    out.write(d)
out.close()

wandb.save('alldata.txt', artifact_name='preprocessed-data')

