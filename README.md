# MA_MLforKIMMDY
Code from my Master thesis on creating machine learning model for hydrogen atom transfer activation energies predicitons from amino acid constellations. Most of the code additionally requires the kgcnn package from the AIMat lab (https://github.com/aimat-lab/gcnn_keras)

- input_generation script contains functions to generate model input from .pdb data
- graph_models script contains all the graph models using during the thesis
- training script contains an example workflow for the PaiNN network
- feed_forward_models script contains example code for models used for the feed forwars network tests (Gaussian process regressor, Random Forests, MLP)
- utils script contains additional functions needed for the workflow (custom loss functions etc.)

The code is written in python 3.8.5. 
