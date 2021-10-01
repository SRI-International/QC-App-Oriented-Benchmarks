from numpy.polynomial.polynomial import Polynomial
from numpy.polynomial.polynomial import polyfit
from collections.abc import Iterable
import functools
import math
import random
import numpy as np
import copy

########## Classical math functions

def gaussian_dist(num_state_qubits, mu, sigma=0.3):
    if mu > 1:
        mu = 1
    if mu < 0:
        mu = 0
    if sigma < 1e-3:
        sigma = 1e-3
    
    dist = {}
    normalization = 0.5 * (math.erf((1-mu)/(np.sqrt(2)*sigma)) - math.erf((0-mu)/(np.sqrt(2)*sigma)))

    for i in range(2**num_state_qubits):
        key = bin(i)[2:].zfill(num_state_qubits)
        a = (i)/(2**num_state_qubits)
        b = (i+1)/(2**num_state_qubits)
        dist[key] = 0.5/normalization * (math.erf((b-mu)/(np.sqrt(2)*sigma)) - math.erf((a-mu)/(np.sqrt(2)*sigma)))
    return dist


def linear_dist(num_state_qubits):
    dist = {}
    for i in range(2**num_state_qubits):
        key = bin(i)[2:].zfill(num_state_qubits)
        dist[key] = (2*i+1)/(2**(2*num_state_qubits))
    return dist


def power_f(i, num_state_qubits, power):
    if isinstance(i, Iterable):
        out = []
        for val in i:
            out.append((val / ((2**num_state_qubits) - 1))**power)
        return np.array(out)
    else:
        return (i / ((2**num_state_qubits) - 1))**power
    
    
def estimated_value(target_dist, f):
    avg = 0
    for key in target_dist.keys():
        x = int(key,2)
        avg += target_dist[key]*f(x)
    return avg
    
    
    
def zeta_from_f(i, func, epsilon, degree, c):
    """
    Intermediate polynomial derived from f to serve as angle for controlled Ry gates.
    """
    rad = np.sqrt(c*(func(i) - 0.5) + 0.5)
    return np.arcsin(rad)



def simplex(n, k):
    """
    Get all ordered combinations of n integers (zero inclusive) which add up to k; the n-dimensional k simplex.
    """
    if k == 0:
        z = [0]*n
        return [z]
    l = [] 
    
    for p in simplex(n,k-1):
        
        for i in range(n):
            a = p[i]+1
            ns = copy.copy(p)
            ns[i] = a
            if ns not in l:
                l.append(ns)
    return l



def binary_expansion(num_state_qubits, poly):
    """
    Convert a polynomial into expression replacing x with its binary decomposition x_0 + 2 x_1 + 4 x_2 + ... 
    
    Simplify using (x_i)^p = x_i for all integer p > 0 and collect coefficients of equivalent expression
    
    """
    n = num_state_qubits
    if isinstance(poly, Polynomial):
        poly_c = poly.coef
    else:
        poly_c = poly
        
    out_front = {}
    out_front[()] = poly_c[0]
    for k in range(1,len(poly_c)):
        for pow_list in simplex(n,k):
            two_exp, denom, t = 0, 1, 0
            for power in pow_list:
                two_exp += t*power
                denom *= np.math.factorial(power)
                t+=1
            nz = np.nonzero(pow_list)[0]
            key = tuple(nz)
            if key not in out_front.keys():
                out_front[key] = 0
            out_front[key] += poly_c[k]*((np.math.factorial(k) / denom) * (2**(two_exp)))
    return out_front


def starting_regions(num_state_qubits):
    """
    For use in bisection search for state preparation subroutine. Fill out the necessary region labels for num_state_qubits.    
    """
    sub_regions = []
    sub_regions.append(['1'])
    for d in range(1,num_state_qubits):
        region = []
        for i in range(2**d):
            key = bin(i)[2:].zfill(d) + '1'
            region.append(key)
        sub_regions.append(region)
        
    return sub_regions



def region_probs(target_dist, num_state_qubits):
    """
    Fetch bisected region probabilities for the desired probability distribution {[p1], [p01, p11], [p001, p011, p101, p111], ...}.
    """
    
    regions = starting_regions(num_state_qubits)
    probs = {}
    n = len(regions)
    for k in range(n):
        for string in regions[k]:
            p = 0
            b = n-k-1
            for i in range(2**b):
                subkey = bin(i)[2:].zfill(b)
                if b == 0:
                    subkey = ''
                try:
                    p += target_dist[string+subkey]
                except KeyError:
                    pass
            probs[string] = p
    return probs


def mc_dist(num_counting_qubits, exact, c_star, method):
    """
    Creates the probabilities of measurements we should get from the phase estimation routine
    
    Taken from Eq. (5.25) in Nielsen and Chuang
    """
    # shift exact value into phase phi which the phase estimation approximates
    if method == 1:
        unshifted_exact = ((exact - 0.5)*c_star) + 0.5
    elif method == 2:
        unshifted_exact = exact
    phi = np.arcsin(np.sqrt(unshifted_exact))/np.pi 

    dist = {}
    precision = int(num_counting_qubits / (np.log2(10))) + 2
    for b in range(2**num_counting_qubits):
        
        # Eq. (5.25), gives probability for measuring an integer (b) after phase estimation routine
        # if phi is too close to b, results in 0/0, but acutally should be 1
        if abs(phi-b/(2**num_counting_qubits)) > 1e-6:
            prob = np.abs(((1/2)**num_counting_qubits) * (1-np.exp(2j*np.pi*(2**num_counting_qubits*phi-b))) / (1-np.exp(2j*np.pi*(phi-b/(2**num_counting_qubits)))))**2
        else:
            prob = 1.0
        
        # calculates the predicted expectation value if measure b in the counting qubits
        a_meas = pow(np.sin(np.pi*b/pow(2,num_counting_qubits)),2)
        if method == 1:
            a = ((a_meas - 0.5)/c_star) + 0.5
        elif method == 2:
            a = a_meas
        a = round(a, precision)

        # generates distribution of expectation values and their relative probabilities
        if a not in dist.keys():
            dist[a] = 0
        dist[a] += prob
    
    return dist


def value_and_max_prob_from_dist(dist):
    """
    Returns the max probability and value from a distribution:
    Ex: From: {0.0: 0.1, 0.33: 0.4, 0.66: 0.2, 1.0: 0.3}
        Returns: (0.33, 0.4)
    """
    value = max(dist, key = dist.get)
    max_prob = dist[value]

    return value, max_prob