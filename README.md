# Recovering Barabási-Albert Parameters of Graphs through Disentanglement

This repo contains the code referred to in the paper [ADD ARXIV LINK].

Below you will find the details for setting up and running the codes for the following 
models:

1. Random Forest predictor 
2. Supervised GNN
3. LSTM VAE

An additional notebook `MIG_scores.ipynb` is provided for computing the MIG scores for the results of Table 2 in the paper.

You should first install the necessary packages and libraries 
by running the command:

`pip install -r requirements.txt`

Where `requirements.txt` can be found in the root project folder.  

## Random Forest 

The code for generating the data, building the model and producing the plots is in `supervised_training.ipynb`. The support functions are in `generate_data.py` and in  `predict_params.py`. 

## Supervised GNN

The code for generating the data, building and running the experiments can be found in the notebook
`supervised_training.ipynb`. The experiments done in this notebook correspond to the ones explained in Table I from the paper. The external tool functions that we use can be found in  `utils.py`, `multi_graph_cnn_layer.py`,`graph_scoring.py`,`graph_ops.py` and `barabasi_albert.py`. The first two scripts implement GCN and originally come from [keras_dgl](https://github.com/vermaMachineLearning/keras-deep-graph-learning).  As explained in the paper, the experiments we try are the following:


1. Standard BA graphs with fixed number of nodes (50), and varying parameter m (varying uniformly at random from 1 to 49). Set `NB_BARABASI_PARAM = 2` and `EXPERIMENT = "standard"`.
2. Nonlinear BA graphs with alpha varying uniformly at random between 1/3 and 3. Set `NB_BARABASI_PARAM = 3` and `EXPERIMENT = "non-linear"`. The graph generation for this experiment is time-consuming, so we provide the data via a link below.

The architecture implemented for the encoder is defined in the `train_multi_gnn_model` function. Additional functions are included for plotting different graph measures, like Figures 3 and 4 from the paper.

For reproducing our results for experiments 1, set random seed to 0 and use [this](https://drive.google.com/drive/folders/1pTRsN76DZQ0JoqRvgW-s6RBiz19ngquj) dataset for experiment 2. Further details can be found in the notebook itself.

## LSTM VAE

The code for building and running the experiments can be found in the notebook
`vae_lstm.ipynb`. At the top of the notebook you will find the main tool
functions specific to the model. Additional external tool functions are used from 
`utils.py` and `multi_graph_cnn_layer.py` which implement GCN and originally come 
from [keras_dgl](https://github.com/vermaMachineLearning/keras-deep-graph-learning). 
At the bottom you will find the experiments used to produce our results. The entire
notebook runs in less than 30 minutes with the default parameter 
settings for the experiments. A more detailed explanation can be found in the notebook itself. 
