set -e

# setup
rm -rf bucket
python demo_setup.py

# simulate uploading two collection runs to our data library
python sim_collection_run.py col1 0 5000
python sim_collection_run.py col2 5000 8000

# create a dataset from our data library
python create_dataset.py \
  --supercategories vehicle \
  --annotation_types bbox \
  --dataset_name vehicle-boxes \
  --dataset_version v1

# train a model on the the uploaded dataset
python train.py \
  --dataset=vehicle-boxes:v1 \
  --model_type=bbox