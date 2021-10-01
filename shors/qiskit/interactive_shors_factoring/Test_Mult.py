""" This file allows to test the Multiplication blocks Ua. This blocks, when put together as explain in
the report, do the exponentiation. 
The user can change N, n, a and the input state, to create the circuit:
    
 up_reg        |+> ---------------------|----------------------- |+>
                                        |
                                        |
                                        |
                                 -------|---------
                    ------------ |               | ------------
 down_reg      |x>  ------------ |     Mult      | ------------  |(x*a) mod N>
                    ------------ |               | ------------
                                 -----------------       

Where |x> has n qubits and is the input state, the user can change it to whatever he wants
This file uses as simulator the local simulator 'statevector_simulator' because this simulator saves
the quantum state at the end of the circuit, which is exactly the goal of the test file. This simulator supports sufficient 
qubits to the size of the QFTs that are going to be used in Shor's Algorithm because the IBM simulator only supports up to 32 qubits
"""

""" Imports from qiskit"""
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import execute, IBMQ, BasicAer

""" Imports to Python functions """
import math
import time

""" Local Imports """
from qfunctions import create_QFT, create_inverse_QFT
from qfunctions import cMULTmodN


""" Function to properly get the final state, it prints it to user """
""" This is only possible in this way because the program uses the statevector_simulator """
def get_final(results, number_aux, number_up, number_down):
    i=0
    """ Get total number of qubits to go through all possibilities """
    total_number = number_aux + number_up + number_down
    max = pow(2,total_number)   
    print('|aux>|top_register>|bottom_register>\n')
    while i<max:
        binary = bin(i)[2:].zfill(total_number)
        number = results.item(i)
        number = round(number.real, 3) + round(number.imag, 3) * 1j
        """ If the respective state is not zero, then print it and store the state of the register where the result we are looking for is.
        This works because that state is the same for every case where number !=0  """
        if number!=0:
            print('|{0}>|{1}>|{2}>'.format(binary[0:number_aux],binary[number_aux:(number_aux+number_up)],binary[(number_aux+number_up):(total_number)]),number)
            if binary[number_aux:(number_aux+number_up)]=='1':
                store = binary[(number_aux+number_up):(total_number)]
        i=i+1

    print(' ')

    return int(store, 2)

""" Main program """
if __name__ == '__main__':

    """ Select number N to do modN"""
    N = int(input('Please insert integer number N: '))
    print(' ')

    """ Get n value used in QFT, to know how many qubits are used """
    n = math.ceil(math.log(N,2))

    """ Select the value for 'a' """
    a = int(input('Please insert integer number a: '))
    print(' ')

    """ Please make sure the a and N are coprime"""
    if math.gcd(a,N)!=1:
        print('Please make sure the a and N are coprime. Exiting program.')
        exit()

    print('Total number of qubits used: {0}\n'.format(2*n+3))

    print('Please check source file to change input quantum state. By default is |2>.\n')

    ts = time.time()
    
    """ Create quantum and classical registers """
    aux = QuantumRegister(n+2)
    up_reg = QuantumRegister(1)
    down_reg = QuantumRegister(n)

    aux_classic = ClassicalRegister(n+2)
    up_classic = ClassicalRegister(1)
    down_classic = ClassicalRegister(n)

    """ Create Quantum Circuit """
    circuit = QuantumCircuit(down_reg , up_reg , aux, down_classic, up_classic, aux_classic)

    """ Initialize with |+> to also check if the control is working"""
    circuit.h(up_reg[0])

    """ Put the desired input state in the down quantum register. By default we put |2> """
    circuit.x(down_reg[1])
    
    """ Apply multiplication""" 
    cMULTmodN(circuit, up_reg[0], down_reg, aux, int(a), N, n)

    """ show results of circuit creation """
    create_time = round(time.time()-ts, 3)
    
    if n < 8: print(circuit)
    
    print(f"... circuit creation time = {create_time}")
    ts = time.time()
    
    """ Simulate the created Quantum Circuit """
    simulation = execute(circuit, backend=BasicAer.get_backend('statevector_simulator'),shots=1)
    
    """ Get the results of the simulation in proper structure """
    sim_result=simulation.result()

    """ Get the statevector of the final quantum state """
    outputstate = sim_result.get_statevector(circuit, decimals=3)

    """ Show the final state after the multiplication """
    after_exp = get_final(outputstate, n+2, 1, n)

    """ show execution time """
    exec_time = round(time.time()-ts, 3)
    print(f"... circuit execute time = {exec_time}")
    
    """ Print final quantum state to user """
    print('When control=1, value after exponentiation is in bottom quantum register: |{0}>'.format(after_exp))