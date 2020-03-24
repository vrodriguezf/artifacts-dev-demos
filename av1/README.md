# av artifacts dev demo scripts

These scripts don't train real models, but are an example of what a simple autonomous vehicle project might look like. There is a train step that given a dataset can produce different types of models, and an eval step to evaluate a trained model on a golden set.

## create_dataset.py

Generate a new "dataset" artifact. This will create a folder with the artifact contents
save the artifact to W&B.

For this example, the contents just contain a fake URL.

E.g.

```python create_dataset.py --datadir=train_data --n_examples 100014```

## train.py

Train a model on a dataset.

```python train.py --datadir train_data --model_type stopsign```

This will make a new model of the specified type.

## train-many.py

```python train-many.py --datadir train_data```

Trains a few example models

## eval.py

First generate a golden dataset:

```python create_dataset.py --datadir=test_data --n_examples 5001 --golden```

Evaluates a given model, using the `dataset:golden` dataset.

```
python eval.py --model 'model-stopsign:latest'
```

## eval-many.py

```
python eval-many.py
```

Evaluates a few example models.