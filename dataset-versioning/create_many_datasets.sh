set -e

python create_dataset.py \
  --supercategories vehicle \
  --annotation_types bbox \
  --dataset_name "vehicle boxes (small)" \
  --dataset_version v1 \
  --select_fraction 0.1

python create_dataset.py \
  --supercategories appliance \
  --annotation_types bbox \
  --dataset_name "appliance boxes" \
  --dataset_version v1

python create_dataset.py \
  --supercategories food \
  --annotation_types segmentation \
  --dataset_name "food segmentation" \
  --dataset_version v1

python create_dataset.py \
  --supercategories furniture \
  --annotation_types bbox \
  --dataset_name "furniture boxes" \
  --dataset_version v1

python create_dataset.py \
  --supercategories animal \
  --annotation_types bbox \
  --dataset_name "animal boxes" \
  --dataset_version v1

python create_dataset.py \
  --supercategories person \
  --annotation_types bbox \
  --dataset_name "people boxes (small)" \
  --select_fraction 0.1 \
  --dataset_version v1

python create_dataset.py \
  --supercategories person \
  --annotation_types bbox \
  --dataset_name "people boxes (medium)" \
  --select_fraction 0.3 \
  --dataset_version v1

python create_dataset.py \
  --supercategories person \
  --annotation_types bbox \
  --dataset_name "people boxes (large)" \
  --dataset_version v1

python create_dataset.py \
  --supercategories food \
  --categories "traffic light" "car" \
  --annotation_types bbox \
  --dataset_name "traffic lights & cars, bounding boxes (sampled)" \
  --select_fraction 0.1 \
  --dataset_version v1