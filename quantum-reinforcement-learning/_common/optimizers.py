'''
Quantum Fourier Transform Benchmark Program - Optimizers
(C) Quantum Economic Development Consortium (QED-C) 2025.
'''

import numpy as np
import math

class Adam:
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        """
        params: list of floats, your initial parameters
        lr: learning rate α
        beta1, beta2: decay rates for the first and second moment estimates
        eps: small constant to avoid division by zero
        """
        self.params = params[:]         # copy of parameter values
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.m = [0.0] * len(params)    # 1st moment
        self.v = [0.0] * len(params)    # 2nd moment
        self.t = 0                      # timestep
        self.grads = [0.0] * len(params) # gradient parameters

    def step(self, grad_fn):
        """
        grads: list of gradients of the same length as self.params
        Returns the updated parameters.
        """
        self.t += 1
        updated = []
        for i in range(len(self.params)):
            self.grads[i] = grad_fn(i)
        
        for i, (p, g) in enumerate(zip(self.params, self.grads)):
            # Update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            # Update biased second raw moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g * g)

            # Compute bias-corrected moments
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            # Parameter update
            p_new = p - self.lr * m_hat / (math.sqrt(v_hat) + self.eps)
            updated.append(p_new)

        self.params = updated
        return self.params

class SPSA:
    def __init__(
        self,
        x0,
        a=0.01,
        c=0.01,
        alpha=0.602,
        gamma=0.101,
        A=0,
        seed=None,
    ):
        """
        x0     : initial parameter vector (array-like)
        a, c   : scale factors for the gain sequences
        alpha  : decay exponent for a_k (usually ~0.602)
        gamma  : decay exponent for c_k (usually ~0.101)
        A      : stability constant (often set to 0 or 10% of max_iter)
        seed   : for reproducible ±1 draws
        """
        self.params = np.array(x0, dtype=float)
        self.a = a
        self.c = c
        self.alpha = alpha
        self.gamma = gamma
        self.A = A
        self.k = 0
        if seed is not None:
            np.random.seed(seed)

    def step(self, loss_fn):
        """
        Perform one SPSA update.

        loss_fn: function taking a parameter vector θ and returning a scalar loss.
        
        Returns the updated parameter vector.
        """
        self.k += 1
        # gain sequences
        ak = self.a / ((self.k + self.A) ** self.alpha)
        ck = self.c / (self.k ** self.gamma)

        # simultaneous perturbation vector Δ ∈ {+1,−1}^d
        delta = np.random.choice([1.0, -1.0], size=self.params.shape)

        # evaluate loss at θ ± c_k Δ
        theta_plus  = self.params + ck * delta
        theta_minus = self.params - ck * delta
        y_plus      = loss_fn(theta_plus)
        y_minus     = loss_fn(theta_minus)

        # gradient estimate
        g_hat = (y_plus - y_minus) / (2.0 * ck) * delta

        # parameter update
        self.params = self.params - ak * g_hat
        return self.params