import os
import random
import wandb

n_epochs = 10

api = wandb.Api()
run = wandb.init(job_type='eval')

preprocessed = api.artifact_version('shawn/artifacts2/nulewc8cg')
run.log_input(preprocessed)
preprocessed_dir = preprocessed.download()
data = open(os.path.join(preprocessed_dir, 'alldata.txt')).read()

model = api.artifact_version('shawn/artifacts2/oq8406rmm')
run.log_input(model)
model_dir = model.download()
model = open(os.path.join(model_dir, 'model.txt')).read()

wandb.log({'test_acc': random.random()})
wandb.log({'test_acc': random.random()})
wandb.log({'test_acc': random.random()})
