# Dataset versioning with W&B Artifacts

This folder contains an end-to-end example of how to version your datasets using W&B Artifacts.

TODO: More info

# setup

First run these setup steps.

```
$ sh download_coco_val.sh

# Special step before release: install wandb from client repo artifacts/next branch
# (PYENV_VERSION=<target_pyenv> pip install -e . in wandb/client folder)

$ pip install -r requirements.txt
$ python demo_setup.py
```

# initialize a wandb project in this folder

```
$ wandb init
```

# Add data to your data library

```
$ python sim_collection_run.py col1 0 5000
$ python sim_collection_run.py col2 5000 8000
```

# create a dataset artifact based

```
python create_dataset.py \
  --supercategories vehicle \
  --annotation_types bbox \
  --dataset_name vehicle-boxes \
  --dataset_version v1
```

# train a model based on the dataset

```
python train.py \
  --dataset=vehicle-boxes:v1 \
  --model_type=bbox
```