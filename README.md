# AI4AM
ARL  Dataset and demo code

### Dependencies
`pip install tensorflow (tensorflow 2.4+)`
`pip install tensorflow-datasets`
`pip install tensorflow-addons`
`pip install numpy`

## Dataset Building
```bash
export TFDS_DATA_DIR=/path/to/data/dir
cd /path/to/repo
tfds build AM_plastic_defects
```

First, we need to specify a location to store the dataset that we are going to download. By default, this points to `~/tensorflow_datasets/`. 

`export TFDS_DATA_DIR=/path/to/dir`

Download and prepare the dataset by using the tensorflow dataset tool:

`tfds build AM_plastic_defects`

Once tensorflow has finished downloading and preparing the dataset, we can check that the dataset is properly set up with a line of python:
```python
import tensorflow_datasets as tfds
tfds.load('am_plastic_defects')
```

## Running Model
`python models/run.py --data_dir=$TFDS_DATA_DIR --dataset=name_of_dataset --mode=simclr`

Set `--mode=simclr` for pretraining with simclr framework, `--model=control_vector` for pretraining with control vector learning.

## Experiment with data augmentations (image transformations)
To experiment with different image transformations, add desired transforms to `models/data_utils.py` and make sure to specify the combinations of transforms in respective `*_transforms` functions.

Example for random crop + random flip:
```python
def pretrain_transforms(image): 
    image = random_crop_and_resize(image, 0.1)
    image = random_flip(image)
    return image
```

## Flags
`--mode`: specify the pretraining mode in ['simclr', 'control_vector'].
`--dataset`: dataset name.
`--data_dir`: specify the directory in which data for `dataset` is stored.
`--momentum`: Momuntum.
`--img_size`: Size to resize input images to.
`--temp`: Temperature parameter.
`--save_model`: Whether to save model, default `False`.
`--model_dir`:  Directory to save model.
`--ckpt`:  Path to load checkpoint, default to `None`.
`--pretrain_epochs`:  Epochs for pretraining. 
`--finetune_epochs`:  Epochs for finetuning.
`--lineareval_epochs`:  Epochs for linear evaluation.
`--pretrain_bs`:  Batch size for pretraining.
`--finetune_bs`:  Batch size for finetuning.
`--lineareval_bs`:  Batch size for linear evaluation.
`--eval_bs`:  Batch size for evaluation.
`--debug`:  Debug mode will increase verbosity, default to `False`.