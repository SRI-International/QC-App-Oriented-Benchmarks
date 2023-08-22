# python file to generate random energies for the H-Lattice model and store them in a json file
import glob
import numpy as np
import random
import sys
import os
from qiskit.opflow.primitive_ops import PauliSumOp
import json



# get the absolute path of the current file
current_folder_path = os.path.dirname(os.path.abspath(__file__))

# add hydrogen lattice python files relative to the current path
sys.path[1:1] = [ "../../qiskit/", "../../_common"]

# import the hydrogen lattice module
import hydrogen_lattice_benchmark as hlb
import common

"""
USER INPUTS
"""

# define the minimum and maximum number of qubits here
# if min_qubits is equal to max_qubits equal to zero, then the code will iterate over all files in the instance folder
min_qubits = 2
max_qubits = 16
average_over = 1000

"""
END OF USER INPUTS
"""


# define a class that has two methods, one two set the number of qubits and other to return the counts for a given circuit
class ResultSimulator(object):

    def __init__(self):
        super().__init__()
        self.num_qubits = 0

    def set_num_qubits(self, num_qubits):
        self.num_qubits = num_qubits

    # method that creates a dictionary with keys as the binary strings and values as the counts assigned randomly
    def get_counts(self, shots=1000):
        
        counts = dict()
        for i in range(shots):
            binary_string = ""
            for j in range(self.num_qubits):
                binary_string += str(random.randint(0, 1))
            if binary_string in counts:
                counts[binary_string] += 1
            else:
                counts[binary_string] = 1
        return counts
    
    # destructor for the class
    def __del__(self):
        pass

# method to store a dictionary of energies in a json file
def store_energies(energy_dict:dict):
    
        # get the absolute path of the current file
        current_folder_path = os.path.dirname(os.path.abspath(__file__))
    
        # get the path of the energy file
        energy_file_path = os.path.join(current_folder_path, "precomputed_random_energies.json")

        # if the file does not exist, create a new file and write the energies
        if not os.path.exists(energy_file_path):
            with open(energy_file_path, "w") as energy_file:
                json.dump(energy_dict, energy_file, indent=4)
        # if the file exists, read the energies from the file and update the energies
        else:
            with open(energy_file_path, "r") as energy_file:
                energy_dict_old = json.load(energy_file)
                energy_dict_old.update(energy_dict)
            with open(energy_file_path, "w") as energy_file:
                json.dump(energy_dict_old, energy_file, indent=4)

        # close the file
        energy_file.close()


# main code starts here

print("Generating random energies for the H-Lattice model and storing them in a json file")

# iterate over all files in the isntance folder if min_qubits is equal to max_qubits
if min_qubits == max_qubits == 0:
    # read all json file names in the instance folder
    instance_filepath_list = [file \
        for file in glob.glob(os.path.join(os.path.dirname(__file__), "..", \
        common.INSTANCE_DIR, f"h*_*_*_*.json"))]
    
    # get all file names from the instnacefilepath list
    filename_list = [os.path.basename(file) for file in instance_filepath_list]

    # get the number of qubits from the file names
    qubit_count_list = [int(filename.split(".")[0].split("_")[0][1:]) for filename in filename_list]
    
    # min and max qubits equal to minimum and maximum of the qubit count list
    min_qubits = min(qubit_count_list)
    max_qubits = max(qubit_count_list)

for qubit_count in np.arange(min_qubits, max_qubits + 2, 2):

    print(f"Generating random energies for {qubit_count} qubits")

    instance_filepath_list = [file \
        for file in glob.glob(os.path.join(os.path.dirname(__file__), "..", \
        common.INSTANCE_DIR, f"h{qubit_count:03}_*_*_*.json"))]
    

    # create a dictionary to store the energies
    energy_dict = dict()
    
    # iterate over all the files in the instance folder
    for instance_filepath in instance_filepath_list:

        ops,coefs = common.read_paired_instance(instance_filepath)
        operator = PauliSumOp.from_list(list(zip(ops, coefs)))
        qc_array, frmt_obs, params = hlb.HydrogenLattice(num_qubits = qubit_count,operator = operator, thetas_array= None, parameterized= None)



        # run the part of code multiple times to average over the results

        average_energy = 0

        for i in range(average_over):

            # create three instance of the ResultSimulator class
            result = []

            for i in range(3):

                res = ResultSimulator()


                # set the number of qubits in the Resultsimulator
                res.set_num_qubits(qubit_count)

                # add the res to the result array
                result.append(res)
                # destroy the res object
                del res


            # compute the energy for this result
            energy = hlb.compute_energy(result_array=result, formatted_observables = frmt_obs, num_qubits=qubit_count )
            average_energy += energy
        
        average_energy /= average_over
        
        # store the energy in the dictionary
        # get the name of the file from the path
        filename = os.path.basename(instance_filepath)
        filename = filename.split(".")[0]
        energy_dict[filename] = average_energy


    # store the energies in a json file
    store_energies(energy_dict)
        
    print("stored energy for ", qubit_count, " qubits")

print("Process completed. Please find the energies in the file precomputed_random_energies.json")

    
