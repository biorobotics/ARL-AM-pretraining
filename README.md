# AI4AM
ARL AI4AM Dataset and demo code

### Dependencies
`pip install tensorflow  tensorflow_datasets`

## Dataset Building
Real 3d print data: `tfds build ai4AM`

Shitty simulated fluid data: `tfds build taichiSim`

## Running a model
`cd models && python ai4am_resnet18_train.py`
