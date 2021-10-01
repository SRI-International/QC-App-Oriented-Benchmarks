""" This file allows to test the calculations done after creating the quantum circuit
It implements the continued fractions to find a possible r and then:
    If r found is odd, tries next approximation of the continued fractions method
    If r found is even, tries to get the factors of N and then:
        If the factors are found, exits
        If the factors are not found, asks user if he wants to continue looking (only if already tried too many times it exits automatically)
"""

from cfunctions import get_factors


""" Main program """
if __name__ == '__main__':

    print('Forcing a case of the AP3421 lectures, with N=143, a=5, x_value=1331 and 11 qubits used to get x_value.')
    print('Check main in source file to change\n')
    
    """ These numbers can be changed to check the desired case. The values that are here by default is to test 
        the case of the slide 28 of lecture 10 of the AP3421 course. The fucntion get_dactors does the continued
        fractions for ( x_value / ( 2^(number_qubits_used_to_get_x_value) ) )
    """
    N = 143
    a = 5
    number_qubits_used_to_get_x_value = 11
    x_value = 1331
    # To check the case of slide 27 of lecture 10 of the AP3421 course, just change x_value to 101

    d=get_factors(int(x_value),int(number_qubits_used_to_get_x_value),int(N),int(a))