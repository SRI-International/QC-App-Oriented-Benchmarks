import numpy as np
from scipy.sparse import dok_array

class HamiltonianCircuitProxy(object):

    h = None
    J = None
    num_qubits = None
    sampler = None
    embedding = None

    def __init__(self):
        self.h = None
        self.J = None
        self.sampler = None
        self.embedding = None

    @property
    def H(self):
        n = self.num_qubits
        Ham = dok_array((n, n+1))
        J_ = dok_array((n, n))
        h_ = np.zeros((n,1))
        for key, value in self.J.items():
            J_[key] = value
        for key, value in self.h.items():
            h_[key] = value
        J_ = J_ + J_.T
        J_ /= 2
        return np.hstack([h_, J_.todense()])