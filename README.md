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
tfds build ai4AM
```

First, we need to specify a location to store the dataset that we are going to download. By default, this points to `~/tensorflow_datasets/`. 

`export TFDS_DATA_DIR=/path/to/dir`

Download and prepare the dataset by using the tensorflow dataset tool:

`tfds build ai4AM`

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
