import os
import wandb
import random

n_epochs = 10

api = wandb.Api()
run = wandb.init(job_type='train')

preprocessed = api.artifact_version('shawn/artifacts5/lonjayo01')
run.log_input(preprocessed)

preprocessed_dir = preprocessed.download()

data = open(os.path.join(preprocessed_dir, 'alldata.txt')).read()

output = open('model.txt', 'w')

for i in range(n_epochs):
    output.write(data)
output.close()

wandb.save('model.txt', artifact_name='model')

wandb.log({'train_acc': random.random()})
wandb.log({'train_acc': random.random()})
wandb.log({'train_acc': random.random()})
