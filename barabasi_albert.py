import networkx as nx
from networkx.generators.classic import empty_graph
import numpy as np
from matplotlib import pyplot as plt

def check_variables(n, m, alpha):
    if m < 1 or m >= n:
        raise nx.NetworkXError(
            f"Barabási–Albert network must have m >= 1 and m < n, m = {m}, n = {n}"
        )
    if alpha < 0 or alpha > 10000:
        raise nx.NetworkXError(
            f"Barabási–Albert network must have 0 <= alpha <= 10000, alpha = {alpha}"
        )

def barabasi_albert_graph(n, m, alpha = 1, seed=None):
    """Returns a random graph according to the Barabási–Albert preferential
    attachment model.

    A graph of $n$ nodes is grown by attaching new nodes each with $m$
    edges that are preferentially attached to existing nodes with high degree.

    Parameters
    ----------
    n : int
        Number of nodes
    m : int
        Number of edges to attach from a new node to existing nodes
    alpha : float
        Coefficient for non-linear preferential attachment
    seed : integer
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    G : Graph

    Raises
    ------
    NetworkXError
        If `m` does not satisfy ``1 <= m < n`` or alpha is negative.

    """
    check_variables(n,m,alpha)
    # Set random seed
    if seed is None:
        seed = np.random.randint(1,100000)
    np.random.seed(seed)
    # Add m initial nodes (m0 in barabasi-speak)
    G = empty_graph(m)
    # Target nodes for new edges
    targets = list(range(m))
    # List of existing nodes, with nodes repeated once for each adjacent edge
    repeated_nodes = []
    # Start adding the other n-m nodes. The first node is m.
    source = m
    while source < n:
        # Add edges to m nodes from the source.
        G.add_edges_from(zip([source] * m, targets))
        # Add one node to the list for each new edge just created.
        repeated_nodes.extend(targets)
        # And the new node "source" has m edges to add to the list.
        repeated_nodes.extend([source] * m)
        # Now choose m unique nodes from the existing nodes
        if alpha == 1:
            # Pick uniformly from repeated_nodes (preferential attachment)
            targets = _random_subset(repeated_nodes, m)
        else:
            # Non-linear preferential attachment
            targets = _alpha_random_subset(repeated_nodes, m, alpha)
        
        source += 1
    return G

def _random_subset(seq, m):
    """ Return m unique elements from seq.

    This differs from random.sample which can return repeated
    elements if seq holds repeated elements.

    """
    targets = set()
    while len(targets) < m:
        x = np.random.choice(seq)
        targets.add(x)
    return targets

def _alpha_random_subset(seq, m, alpha):
    """ Return m unique elements from seq, generated from the distribution
        imposed by alpha.

        This differs from random.sample which can return repeated
        elements if seq holds repeated elements.

    """
    targets = set()
    #sorted = np.sort(seq)
    non_norm_probs = np.bincount(seq)**alpha
    probs = non_norm_probs / sum(non_norm_probs)
    a = len(probs)
    while len(targets) < m:
        x = np.random.choice(a,p=probs)
        #x = np.nonzero(np.random.multinomial(1, probs))[0][0]
        targets.add(x)
    
    return targets

def draw_random_ba_graphs(n,m,alpha,nrows=2,ncols=2):
    graphs = [barabasi_albert_graph(n,m,alpha) for idx in range(nrows*ncols)]
    fig, axes = plt.subplots(nrows=2, ncols=2)
    ax = axes.flatten()
    for i in range(nrows*ncols):
        nx.draw_networkx(graphs[i], ax=ax[i],node_color="orchid",edge_color='grey')
        ax[i].set_axis_off()
    plt.show()

if __name__ == "__main__":
    n = 50
    m = 5
    alpha = 0.5
    seed = 1
    G = barabasi_albert_graph(n, m, seed = seed)
    adj_matrix = nx.linalg.graphmatrix.adjacency_matrix(G, nodelist=None, weight=None)
    print(adj_matrix)
    print('hej')
    G = barabasi_albert_graph(n, m, alpha = alpha, seed = seed)
    adj_matrix = nx.linalg.graphmatrix.adjacency_matrix(G, nodelist=None, weight=None)
    print(adj_matrix)