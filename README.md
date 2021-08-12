# AI4AM
ARL AI4AM Dataset and demo code

### Dependencies
`pip install tensorflow  tensorflow-datasets`

## Dataset Building
```bash
export TFDS_DATA_DIR=/path/to/data/dir
cd /path/to/repo
tfds build ai4AM
```

First, we need to specify a location to store the dataset that we are going to download. By default, this points to `~/tensorflow_datasets/`. 

`export DATA_DIR=/path/to/dir`

Download and prepare the dataset by using the tensorflow dataset tool:

`tfds build ai4AM`

Once tensorflow has finished downloading and preparing the dataset, we can check that the dataset is properly set up with a line of python:
```python
import tensorflow_datasets as tfds
tfds.load('am_plastic_defects')
```


## Running Model
`python models/run.py --data_dir=/path/to/data/dir`


