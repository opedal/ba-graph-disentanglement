import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
import sys
import networkx as nx
import sklearn
from sklearn.metrics import mutual_info_score

class MIG():
    """
    docstring
    """
    def __init__(self,z,v):
        self.z=z
        self.v=v

    def normalize_data(self, data, mean=None, stddev=None):
        if len(data.shape)<2:
            if mean is None : 
                mean = np.mean(data)
            if stddev is None:
                stddev = np.std(data)
            return (data - mean) / stddev, mean, stddev
        else:
            if mean is None :
                mean = np.mean(data, axis=1)
            if stddev is None:
                stddev = np.std(data, axis=1)
            return (data - mean[:, np.newaxis]) / stddev[:, np.newaxis], mean, stddev
    
    def discrete_entropy(self):
        """Compute discrete mutual information."""
        num_factors = self.v.shape[0]
        h = np.zeros(num_factors)
        for j in range(num_factors):
            h[j] = sklearn.metrics.mutual_info_score(self.v[j, :], self.v[j, :])
        return h

    def discretize_data(self,target, num_bins=10):
        """Discretization based on histograms."""
        target = np.nan_to_num(target)
        discretized = np.zeros_like(target)
        if len(target.shape)>1:
            for i in range(target.shape[0]):
                discretized[i, :] = np.digitize(target[i, :], np.histogram(target[i, :], num_bins)[1][:-1])
        else:
            discretized = np.digitize(target, np.histogram(target, num_bins)[1][:-1])
        return discretized

    def compute_mig(self):
        """
        docstring
        """
        #print('start computing MIG score')
        if self.z.shape[0] > 1:
            self.z, z_mean, z_std = self.normalize_data(data=self.z)
            self.v, v_mean, v_std = self.normalize_data(data=self.v)

            z_discrete = self.discretize_data(self.z)
            v_discrete = self.discretize_data(self.v)
            self.z = z_discrete
            self.v = v_discrete


            # m is [num_latents, num_factors]
            m = self.discrete_mutual_info()
            assert m.shape[0] == self.z.shape[0]
            assert m.shape[1] == self.v.shape[0]

            entropy = self.discrete_entropy()
            sorted_m = np.sort(m, axis=0)[::-1]
            mig_score = np.mean(np.divide(sorted_m[0, :] - sorted_m[1, :], entropy[:]))
        else:
            mig_score = "MIG not defined for one latent variable"
        return mig_score

    def discrete_mutual_info(self):
        """Compute discrete mutual information."""
        num_codes = self.z.shape[0]
        num_factors = self.v.shape[0]
        m = np.zeros([num_codes, num_factors])
        for i in range(num_codes):
            for j in range(num_factors):

                if num_factors > 1:
                    m[i, j] = mutual_info_score(self.v[j, :], self.z[i, :])
                elif num_factors == 1:
                    m[i, j] = mutual_info_score(np.squeeze(self.v), self.z[i, :])

        return m
