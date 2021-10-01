"""
This is the final implementation of Shor's Algorithm using the circuit presented in section 2.3 of the report about the second
simplification introduced by the base paper used.
The circuit is general, so, in a good computer that can support simulations infinite qubits, it can factorize any number N. The only limitation
is the capacity of the computer when running in local simulator and the limits on the IBM simulator (in the number of qubits and in the number
of QASM instructions the simulations can have when sent to IBM simulator).
The user may try N=21, which is an example that runs perfectly fine even just in local simulator because, as in explained in report, this circuit,
because implements the QFT sequentially, uses less qubits then when using a "normal"n QFT.
"""

""" Imports from qiskit"""
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import execute, IBMQ
from qiskit import BasicAer

import sys

""" Imports to Python functions """
import math
import array
import fractions
import numpy as np
import time

""" Local Imports """
from cfunctions import check_if_power, get_value_a
from cfunctions import get_factors
from qfunctions import create_QFT, create_inverse_QFT
from qfunctions import getAngle, cMULTmodN


""" Main program """
if __name__ == '__main__':

    """ Ask for analysis number N """   

    N = int(input('Please insert integer number N: '))

    print('input number was: {0}\n'.format(N))
    
    """ Check if N==1 or N==0"""

    if N==1 or N==0: 
       print('Please put an N different from 0 and from 1')
       exit()
    
    """ Check if N is even """

    if (N%2)==0:
        print('N is even, so does not make sense!')
        exit()
    
    """ Check if N can be put in N=p^q, p>1, q>=2 """

    """ Try all numbers for p: from 2 to sqrt(N) """
    if check_if_power(N)==True:
       exit()

    print('Not an easy case, using the quantum circuit is necessary\n')

    """ To login to IBM Q experience the following functions should be called """
    """
    IBMQ.delete_accounts()
    IBMQ.save_account('insert token here')
    IBMQ.load_accounts()"""

    """ Get an integer a that is coprime with N """
    a = get_value_a(N)

    """ If user wants to force some values, can do that here, please make sure to update print and that N and a are coprime"""
    """print('Forcing N=15 and a=4 because its the fastest case, please read top of source file for more info')
    N=15
    a=2"""

    """ Get n value used in Shor's algorithm, to know how many qubits are used """
    n = math.ceil(math.log(N,2))
    
    print('Total number of qubits used: {0}\n'.format(2*n+3))

    ts = time.time()
    
    """ Create quantum and classical registers """

    """auxilliary quantum register used in addition and multiplication"""
    aux = QuantumRegister(n+2)
    """single qubit where the sequential QFT is performed"""
    up_reg = QuantumRegister(1)
    """quantum register where the multiplications are made"""
    down_reg = QuantumRegister(n)
    """classical register where the measured values of the sequential QFT are stored"""
    up_classic = ClassicalRegister(2*n)
    """classical bit used to reset the state of the top qubit to 0 if the previous measurement was 1"""
    c_aux = ClassicalRegister(1)

    """ Create Quantum Circuit """
    circuit = QuantumCircuit(down_reg , up_reg , aux, up_classic, c_aux)

    """ Initialize down register to 1"""
    circuit.x(down_reg[0])

    """ Cycle to create the Sequential QFT, measuring qubits and applying the right gates according to measurements """
    for i in range(0, 2*n):
        """reset the top qubit to 0 if the previous measurement was 1"""
        circuit.x(up_reg).c_if(c_aux, 1)
        circuit.h(up_reg)
        cMULTmodN(circuit, up_reg[0], down_reg, aux, a**(2**(2*n-1-i)), N, n)
        """cycle through all possible values of the classical register and apply the corresponding conditional phase shift"""
        for j in range(0, 2**i):
            """the phase shift is applied if the value of the classical register matches j exactly"""
            circuit.u1(getAngle(j, i), up_reg[0]).c_if(up_classic, j)
        circuit.h(up_reg)
        circuit.measure(up_reg[0], up_classic[i])
        circuit.measure(up_reg[0], c_aux[0])

    """ show results of circuit creation """
    create_time = round(time.time()-ts, 3)
    
    #if n < 8: print(circuit)
    
    print(f"... circuit creation time = {create_time}")
    ts = time.time()
    
    """ Select how many times the circuit runs"""
    number_shots=int(input('Number of times to run the circuit: '))
    if number_shots < 1:
        print('Please run the circuit at least one time...')
        exit()

    """ Print info to user """
    print('Executing the circuit {0} times for N={1} and a={2}\n'.format(number_shots,N,a))

    """ Simulate the created Quantum Circuit """
    simulation = execute(circuit, backend=BasicAer.get_backend('qasm_simulator'),shots=number_shots)
    """ to run on IBM, use backend=IBMQ.get_backend('ibmq_qasm_simulator') in execute() function """
    """ to run locally, use backend=BasicAer.get_backend('qasm_simulator') in execute() function """

    """ Get the results of the simulation in proper structure """
    sim_result=simulation.result()
    counts_result = sim_result.get_counts(circuit)

    """ show execution time """
    exec_time = round(time.time()-ts, 3)
    print(f"... circuit execute time = {exec_time}")
    
    """ Print info to user from the simulation results """
    print('Printing the various results followed by how many times they happened (out of the {} cases):\n'.format(number_shots))
    i=0
    while i < len(counts_result):
        print('Result \"{0}\" happened {1} times out of {2}'.format(list(sim_result.get_counts().keys())[i],list(sim_result.get_counts().values())[i],number_shots))
        i=i+1
    
    """ An empty print just to have a good display in terminal """
    print(' ')

    """ Initialize this variable """
    prob_success=0
    
    """ For each simulation result, print proper info to user and try to calculate the factors of N"""
    i=0
    while i < len(counts_result):

        """ Get the x_value from the final state qubits """
        all_registers_output = list(sim_result.get_counts().keys())[i]
        output_desired = all_registers_output.split(" ")[1]
        x_value = int(output_desired, 2)
        prob_this_result = 100 * ( int( list(sim_result.get_counts().values())[i] ) ) / (number_shots)

        print("------> Analysing result {0}. This result happened in {1:.4f} % of all cases\n".format(output_desired,prob_this_result))

        """ Print the final x_value to user """
        print('In decimal, x_final value for this result is: {0}\n'.format(x_value))

        """ Get the factors using the x value obtained """   
        success = get_factors(int(x_value),int(2*n),int(N),int(a))

        if success==True:
            prob_success = prob_success + prob_this_result

        i=i+1

    print("\nUsing a={0}, found the factors of N={1} in {2:.4f} % of the cases\n".format(a,N,prob_success))