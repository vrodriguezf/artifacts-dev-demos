# av artifacts dev demo scripts

These scripts don't train real models, but are an example of what a simple autonomous vehicle project might look like. There is a train step that given a dataset can produce different types of models, and an eval step to evaluate a trained model on a golden set.

## gen_dataset.py

Generate a new "dataset" in a folder. E.g.

```python create_dataset.py --dir=train_data --n_examples 100014```

## train.py

Train a model on a dataset.

```python train.py --datadir train_data --model_type stopsign```

This will make a new model of the specified type.

## train-many.py

```python train-many.py --datadir train_data```

Trains a few example models

## new_golden_set.py

```
python gen_dataset.py --dir test_data -n_examples 8042
python new_golden_set.py --datadir test_data
```

## eval.py

```
python eval.py --model 'model-stopsign:latest'
```

Evaluates a given model, using the `dataset-test-main:golden` dataset.

## eval-many.py

```
python eval-many.py --model 'model-stopsign:latest'
```

Evaluates a few example models.