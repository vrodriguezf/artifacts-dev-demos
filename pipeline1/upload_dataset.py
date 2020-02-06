import sys
import time
import wandb

run = wandb.init(job_type='upload')

dataset_dir = sys.argv[1]
run.log_artifact('raw-data', dataset_dir)
