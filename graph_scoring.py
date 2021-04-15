import numpy as np

from networkx.algorithms.centrality import betweenness_centrality, closeness_centrality
from networkx.algorithms.connectivity import average_node_connectivity
import networkx as nx
from networkx.algorithms.cluster import triangles

class GraphScorer:

    def __init__(self):
        self._scores = {#'diameter': self.diameter,
                        #'radius': self.radius,
                        #'avg_eccentricity':self.avg_eccentricity,
                        'avg_degree': self.avg_degree,
                        'std_degree': self.std_of_degree,
                        'max_degree': self.max_degree,
                        'min_degree': self.min_degree,
                        'avg_between_centrality': self.avg_betweeness_centrality,
                        'std_between_centrality': self.std_betweeness_centrality,
                        'avg_close_centrality': self.avg_closeness_centrality,
                        'std_close_centrality': self.std_closeness_centrality,
                        'avg_node_connectivity': self.avg_node_connectivity,
                        'num_triangles': self.num_triangles,
                        'max_clique':self.max_clique,
                        #'avg_shortest_path':self.avg_shortest_path
                        }

    def score(self, graph, score_name):
        try:
            score_func = self._scores[score_name]
            return score_func(graph)
        except KeyError:
            print("score not found, try one of: " + " / ".join(self.available_scores()))
            return 0

    def available_scores(self):
        return list(self._scores.keys())

    def num_scores(self):
        return len(self.available_scores())

    def avg_betweeness_centrality(self, graph):
        between_centr = betweenness_centrality(graph)
        return np.average(list(between_centr.values()))

    def std_betweeness_centrality(self, graph):
        between_centr = betweenness_centrality(graph)
        return np.std(list(between_centr.values()))

    def avg_closeness_centrality(self, graph):
        close_centr = closeness_centrality(graph)
        return np.average(list(close_centr.values()))

    def std_closeness_centrality(self, graph):
        close_centr = closeness_centrality(graph)
        return np.std(list(close_centr.values()))

    def avg_node_connectivity(self, graph):
        return average_node_connectivity(graph)

    def diameter(self, graph):
        # returns diameter, i.e. max eccentricity in the graph
        try:
            return nx.diameter(graph)
        except:
            return 0

    def radius(self, graph):
        # returns radius, i.e. min eccentricity in the graph
        return nx.radius(graph)

    def avg_eccentricity(self, graph):
        try:
            return np.average(list(nx.eccentricity(graph).values()))
        except:
            return 0

    def avg_degree(self, graph):
        return np.average([dg for nd, dg in graph.degree()])

    def std_of_degree(self, graph):
        return np.std([dg for nd, dg in graph.degree()])

    def max_degree(self, graph):
        return np.max([dg for nd, dg in graph.degree()])

    def min_degree(self, graph):
        return np.min([dg for nd, dg in graph.degree()])

    def num_triangles(self, graph):
        return sum(triangles(graph).values())/3

    def max_clique(self, graph):
        """
        :return: size of the largest clique in the graph, a.k.a graph_clique_number
        """
        return nx.graph_clique_number(graph)

    def avg_shortest_path(self,graph):
        return nx.average_shortest_path_length(graph)

    def small_world_omega(self,graph):
        """
        Returns the small-world coefficient (omega) of a graph : omega = Lr/L - C/Cl
        where:
        C = average clustering coefficient
        L = average shortest path length of G
        Lr = the average shortest path length of an equivalent random graph
        Cl = the average clustering coefficient of an equivalent lattice graph
        """
        return nx.omega(graph)

    def small_world_sigma(self,graph):
        """
        Returns the small-world coefficient (sigma) of the given graph. sigma = C/Cr / L/Lr
        where:
        C = the average clustering coefficient
        L = the average shortest path length of G
        Cr = the average clustering coefficient
        Lr = average shortest path length of an equivalent random graph.
        """
        return nx.sigma(graph)