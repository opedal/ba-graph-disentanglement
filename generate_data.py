import numpy as np
from numpy.random import randint
import networkx as nx
from networkx.generators import erdos_renyi_graph
from barabasi_albert import barabasi_albert_graph
from utils import preprocess_adj_tensor_with_identity, preprocess_adj_tensor
from sklearn.utils import shuffle
## Progress bar for graph generation
from tqdm import tqdm

def generate_data(num_graphs, n, num_params, sym_norm=True, num_filters=1,):
    A, X, Y = generate_multi_barabasi_graphs(num_graphs=num_graphs, n=n, num_ba_params=num_params)
    # Load and prepare the adjacency matrices
    A = np.array(A)
    num_graph_nodes = A.shape[1]
    num_graphs = int(A.shape[0] / num_graph_nodes)

    A = np.split(A, num_graphs, axis=0)
    A = np.array(A)

    if num_filters == 1:
        A_mod = preprocess_adj_tensor(A, sym_norm)
    elif num_filters == 2:
        A_mod = preprocess_adj_tensor_with_identity(A, sym_norm)
    else:
        print("expected num_filters in [1,2], got ", num_filters)
        return
    # Load and prepare the feature matrices
    X = np.array(X)
    X = np.split(X, num_graphs, axis=0)
    X = np.array(X)

    # Load and prepare the parameter vectors
    Y = np.array(Y)

    # Shuffle data before training
    A, A_mod, X, Y = shuffle(A, A_mod, X, Y)

    return A, A_mod, X, Y

def generate_barabasi_adj_matrix(n, m, alpha,seed=None):
    """Generates a Barabasi Alert graph and returns an adjacency matrix with
     a node ordering defined by the networkx Graph library"""
    networkx_graph = barabasi_albert_graph(n, m, alpha,seed=seed)
    adj_matrix = nx.linalg.graphmatrix.adjacency_matrix(networkx_graph, nodelist=None, weight=None)
    return adj_matrix.todense()

def generate_BA_graphs_for_GNN(nb_graphs, n, train_size,NB_BARABASI_PARAM=2,exp_type = "standard",train=True):
    """Generates 'nb_graphs' Barabasi graphs. Each graph has the same number of nodes 'n', the first 
    parameter of the Barabasi-Albert model. The second parameter 'm' is randomly generated 
    in 1 <= m < n. The third parameter alpha is generated in 1/3 < alpha < 3.
    The function outputs the adjacency matrices, the feature matrices (identity matrices) 
    and the parameters n and m used in the data generation process"""
    
    # Each matrix generated during the graph generation process is vertically concatenated to the output
    adjacency_matrices = np.zeros((nb_graphs*n, n))
    feature_matrices = np.zeros((nb_graphs*n, n))
    parameter_vectors = np.zeros((nb_graphs, NB_BARABASI_PARAM))

    for sample_graph_index in tqdm(range(nb_graphs)):
        
        if train:
            set_seed=sample_graph_index
        else:
            set_seed=sample_graph_index + train_size
            
        np.random.seed(set_seed)
        seed(set_seed)
        
        # Choose m randomly s.t 1 <= m < n
        m = randint(1, n-1)
        
        # Choose alpha according to experiment type
        if exp_type == "standard":
            alpha = 1
        elif exp_type == "alpha noise":
            alpha = np.random.normal(1,0.1)
        elif exp_type == "non-linear":
            alpha = np.random.uniform(1/3, 3)
        elif isinstance(exp_type, str):
            raise ValueError
        else:
            raise TypeError

        # Generate Barabasi Graph adjacency matrix
        adj_matrix = generate_barabasi_adj_matrix(n, m, alpha, seed=set_seed)

        # Update output matrices
        row_start_index = n * sample_graph_index
        
        adjacency_matrices[row_start_index: row_start_index + n, :] = adj_matrix
        feature_matrices[row_start_index: row_start_index + n, :] = np.eye(n)
        if NB_BARABASI_PARAM == 2:
            parameter_vectors[sample_graph_index, :] = [n, m]
        elif NB_BARABASI_PARAM == 3:
            parameter_vectors[sample_graph_index, :] = [n, m, alpha]
    
    return adjacency_matrices, feature_matrices, parameter_vectors

def generate_multi_barabasi_graphs(num_graphs, n, num_ba_params, min_alpha=0,max_alpha=3):
    """Generates 'nb_graphs' Barabasi graphs. Each graph has the same number of nodes 'n', the first
    parameter of the Barabasi-Albert model. The second parameter 'm' is randomly generated
    in 1 <= m < n. The function outputs the adjacency matrices, the feature matrices (identity matrices)
    and the parameters n and m used in the data generation process"""

    # Each matrix generated during the graph generation process is vertically concatenated to the output
    adjacency_matrices = np.zeros((num_graphs * n, n))
    feature_matrices = np.zeros((num_graphs * n, n))
    parameter_vectors = np.zeros((num_graphs, num_ba_params))

    alpha = 1
    assert (num_ba_params == 2 or num_ba_params == 3)

    for sample_graph_index in range(num_graphs):
        # Choose m randomly s.t 1 <= m < n
        m = randint(1, n - 1)

        if num_ba_params == 3:
            alpha = np.random.uniform(min_alpha, max_alpha)

        # Generate Barabasi Graph adjacency matrix
        adj_matrix = generate_barabasi_adj_matrix(n, m, alpha)

        # Update output matrices
        row_start_index = n * sample_graph_index

        adjacency_matrices[row_start_index: row_start_index + n, :] = adj_matrix
        feature_matrices[row_start_index: row_start_index + n, :] = np.eye(n)

        if num_ba_params == 3:     parameter_vectors[sample_graph_index, :] = [n, m, alpha]
        elif num_ba_params == 2:    parameter_vectors[sample_graph_index, :] = [n, m]

    return adjacency_matrices, feature_matrices, parameter_vectors

def generate_ba_nx_graphs(num_graphs, n, num_ba_params, min_alpha=0, max_alpha=3):

    ba_graphs = []
    labels = np.zeros((num_graphs, num_ba_params))
    assert (num_ba_params == 2 or num_ba_params == 3)

    for sample_graph_index in range(num_graphs):
        # Choose m randomly s.t 1 <= m < n
        m = randint(1, n - 1)
        alpha = 1
        if num_ba_params == 3:
            alpha = np.random.uniform(min_alpha, max_alpha)

        ba_graph = barabasi_albert_graph(n,m,alpha)
        ba_graphs.append(ba_graph)

        if num_ba_params == 2: labels[sample_graph_index,:] = [n,m]
        elif num_ba_params == 3: labels[sample_graph_index,:] = [n,m,alpha]

    return ba_graphs, labels

def generate_er_nx_graphs(num_graphs,n):

    er_graphs = []
    for _ in range(num_graphs):
        p = np.random.random()
        er_graphs.append(erdos_renyi_graph(n,p))

    return er_graphs
