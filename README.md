# AI4AM
ARL AI4AM Dataset and demo code

### Dependencies
`pip install tensorflow  tensorflow_datasets`

## Dataset Building
Real 3d print data: `tfds build ai4AM`

Download link: https://drive.google.com/file/d/1KTA_BAdh86Oo3RPrCMeo-zu2o2f_zCWV/view?usp=sharing

Shitty simulated fluid data: `tfds build taichiSim`

## Running a model
`cd models && python ai4am_resnet18_train.py`
