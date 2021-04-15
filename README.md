# Disentangling Parameters of Sequential Graph Generator Models

Below you will find the details for setting up and running the codes for the following 
models:

1. Vanilla GCN encoder. 
2. LSTM AE.
3. LSTM VAE.
4. CGVAE adpated from Liu's [paper](https://arxiv.org/abs/1805.09076).

An additional notebook `MIGs_CGVAE_LSTM.ipynb` is provided for computing the MIG scores for the *Disentanglement* results subsection.

You should first install the necessary packages and libraries 
by running the command:

`pip install -r requirements.txt`

Where `requirements.txt` can be found in the root project folder.  

## Random Forest 

The code for generating the data, building the model and producing the plots is in `supervised_training.ipynb`. The support functions are in `generate_data.py` and in  `predict_params.py`. 

## GCN encoder

The code for generating the data, building and running the experiments can be found in the notebook
`supervised_training.ipynb`. The experiments done in this notebook correspod to the ones explained in Table I from the paper. The external tool functions that we use can be found in  `utils.py`, `multi_graph_cnn_layer.py`,`graph_scoring.py`,`graph_ops.py` and `barabasi_albert.py`. The first two scripts implement GCN and originally come from [keras_dgl](https://github.com/vermaMachineLearning/keras-deep-graph-learning).  As explained in the paper, the experiments we try are the following:


1. Standard BA graphs with fixed number of nodes (50), and varying parameter m (varying uniformly at random from 1 to 49)
2. Standard BA graphs with gaussian noise on parameter alpha, with standard deviation of 0.1. 
3. Nonlinear BA graphs with alpha varying uniformly at random between 1/3 and 3. The graph generation for this experiment is computationally expensiveo (more than 2 hours).

The architecture implemented for the encoder is defined in the `train_multi_gnn_model` function. Additional functions are included for plotting different graph measures, like Figures 5 and 6 from the paper.

For reproducing our results for experiments 1 and 2, set random seed to 0 and use [this](https://drive.google.com/drive/folders/1pTRsN76DZQ0JoqRvgW-s6RBiz19ngquj) dataset for experiment 3. Further details can be found in the notebook itself.

## LSTM AE 

The code for building and running the experiments can be found in the notebook
`ae_lstm.ipynb`. At the top of the notebook you will find the main tool
functions specific to the model. Additional external tool functions are used from 
`utils.py` and `multi_graph_cnn_layer.py` which implement GCN and originally come 
from [keras_dgl](https://github.com/vermaMachineLearning/keras-deep-graph-learning). 
At the bottom you will find the experiments used to produce our results. The entire
notebook runs in less than 30 minutes with the current parameter settings 
for the experiments. More detailed explanation can be found in the notebook itself. 

## LSTM VAE

It runs in the same way as for LSTM AE. The notebook can be found in `vae_lstm.ipynb`.
Again it should run in less than 30 minutes with the default parameter 
settings for the experiments.  

## CGVAE

To run CGVAE you have to go into the CGVAE_adaptation folder, and then
1) run source install.sh. This will set up a virtual environment and install the requirements, including tensorflow 1.3.0. Note that we used tensorflow 2 for the rest of the project. 
2) Once the environment is set up, go into the data folder and run get_ba.py to generate the training data. 

```
python get_ba.py

```
3) To train the CGVAE model on the generated graphs run  


```
python CGVAE.py

```
This will also generate a file called `latent_vars.csv`, where the latent means and logvariances from the last epoch are saved, together with the generative parameters for each graph. These are then used for calculating the MIG. 

To run experiments with different hyperparameters, run `run_experiments.py`, specifying the hyperparameters. For example: 

```
python run_experiments.py --dataset ba --batch_size 32 --num_epochs 10 --hidden_dim 5 --lr 0.1 --kl_tradeoff 0.5 --optstep 0 

```
