"""
This file contains various helper functions for the various Shor's algorithm benchmarks,
including order finding and factoring.
"""
import math
from math import gcd
import numpy as np
############### Data Used in Shor's Algorithm Benchmark

# Array of prime numbers used to construct numbers to be factored
primes = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89,
          97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181,
          191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281,
          283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397,
          401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503,
          509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619,
          631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701, 709, 719, 727, 733, 739, 743,
          751, 757, 761, 769, 773, 787, 797, 809, 811, 821, 823, 827, 829, 839, 853, 857, 859, 863,
          877, 881, 883, 887, 907, 911, 919, 929, 937, 941, 947, 953, 967, 971, 977, 983, 991, 997]


# Function to generate numbers for factoring
def generate_numbers():
    # Numbers available for factoring (multiples of primes) indexed by number of qubits to use
    numbers = [None] * 21

    for f1 in primes:
        for f2 in primes:
            if f1 >= f2: continue
            number = f1 * f2
            idx = int(math.ceil(math.log(number, 2)))
            if idx >= len(numbers): continue
            if numbers[idx] == None: numbers[idx] = []
            # if len(numbers[idx]) > 20: continue         # limit to 20 choices
            # don't limit, as it skews the choices
            numbers[idx].append(number)
    return numbers

############### General Functions for factoring analysis

# Verifies base**order mod number = 1
def verify_order(base, number, order):
    return base ** order % number == 1

# Generates the base for base**order mod number = 1
def generate_base(number, order):
    # Max values for a and x
    a = number
    x = math.log(number ** order - 1, number)

    # a must be less than number
    while a >= number or not verify_order(a, number, order):
        a = int((number ** x + 1) ** (1 / order))
        x -= 1 / a
    return a

# Choose a base at random < N / 2 without a common factor of N
def choose_random_base(N):
    # try up to 100 times to find a good base
    for guess in range(100):
        a = int(np.random.random() * (N / 2))
        if gcd(a, N) == 1:
            return a

    print(f"Ooops, chose non relative prime {a}, gcd={gcd(a, N)}, giving up ...")
    return 0


# Determine factors from the period
def determine_factors(r, a, N):
    # try to determine the factors
    if r % 2 != 0:
        r *= 2

    apowrhalf = pow(a, r >> 1, N)

    f1 = gcd(apowrhalf + 1, N)
    f2 = gcd(apowrhalf - 1, N)

    # if this f1 and f2 are not the factors
    # and not both 1
    # and if multiplied together divide N evenly
    # --> then try multiplying them together and dividing into N to obtain the factors
    f12 = f1 * f2
    if ((not f12 == N)
            and f12 > 1
            and int(1. * N / (f12)) * f12 == N):
        f1, f2 = f12, int(N / (f12))

    return f1, f2

############### Functions for Circuit Derivation

def egcd(a, b):
    if a == 0:
        return (b, 0, 1)
    else:
        g, y, x = egcd(b % a, a)
        return (g, x - (b // a) * y, y)

def modinv(a, m):
    g, x, y = egcd(a, m)
    if g != 1:
        raise Exception('modular inverse does not exist')
    else:
        return x % m

# TODO: Merge the following Angle functions or change the names
# Function that calculates the angle of a phase shift in the sequential QFT based on the binary digits of a.
# a represents a possible value of the classical register
def getAngle(a, n):
    #convert the number a to a binary string with length n
    s=bin(int(a))[2:].zfill(n)
    angle = 0
    for i in range(0, n):
        # if the digit is 1, add the corresponding value to the angle
        if s[n-1-i] == '1':
            angle += math.pow(2, -(n-i))
    angle *= np.pi
    return angle

# Function that calculates the array of angles to be used in the addition in Fourier Space
def getAngles(a,n):
    #convert the number a to a binary string with length n
    s=bin(int(a))[2:].zfill(n)
    angles=np.zeros([n])
    for i in range(0, n):
        for j in range(i,n):
            if s[j]=='1':
                angles[n-i-1]+=math.pow(2, -(j-i))
        angles[n-i-1]*=np.pi
    return angles