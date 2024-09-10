'''
Hamiltonian Simulation Benchmark Program - HamLib Utility Functions
(C) Quantum Economic Development Consortium (QED-C) 2024.
'''

'''
DEVNOTE: TODO - Describe what functions are in here ... 
'''

import h5py
import re
import os
import requests
import zipfile
import json
from dataclasses import dataclass

verbose = False


@dataclass
class HamLibData:
    name: str
    file_name: str
    url: str


_base_url = 'https://portal.nersc.gov/cfs/m888/dcamps/hamlib/'


hamiltonians = [
    HamLibData('TFIM', 'tfim.hdf5', f'{_base_url}condensedmatter/tfim/tfim.zip'),
    HamLibData('Fermi-Hubbard-1D', 'FH_D-1.hdf5', f'{_base_url}condensedmatter/fermihubbard/FH_D-1.zip'),
    HamLibData('Bose-Hubbard-1D', 'BH_D-1_d-4.hdf5', f'{_base_url}condensedmatter/bosehubbard/BH_D-1_d-4.zip'),
    HamLibData('Heisenberg', 'heis.hdf5', f'{_base_url}condensedmatter/heisenberg/heis.zip'),
    HamLibData('Max3Sat', 'random_max3sat-hams.hdf5', f'{_base_url}binaryoptimization/max3sat/random/random_max3sat-hams.hdf5.zip')
]

def create_full_filenames(hamiltonian_name):
    """
    Fetches the filename for a given Hamiltonian.
    If the name is not present in the dictionary, append '.hdf5' and return.
    
    Args:
    hamiltonian_name (str): The name of the Hamiltonian to lookup.

    Returns:
    str: The filename associated with the Hamiltonian.
    """

    ham_names = [h.name for h in hamiltonians]

    if hamiltonian_name in ham_names:
        return hamiltonians[ham_names.index(hamiltonian_name)].file_name
    else:
        return hamiltonian_name + '.hdf5'
    
def extract_dataset_hdf5(filename, dataset_name):
    """
    Extract a dataset from an HDF5 file.

    Args:
        filename (str): The path to the HDF5 file.
        dataset_name (str): The name of the dataset to extract.

    Returns:
        numpy.ndarray or None: The extracted dataset as a NumPy array, or None if the dataset is not found.

    """
    # Open the HDF5 file
    with h5py.File(filename, 'r') as file:
        if dataset_name in file:
            dataset = file[dataset_name]
            data = dataset[()] if dataset.shape == () else dataset[:]
        else:
            data = None
            if verbose:
                print(f"Dataset {dataset_name} not found in the file.")
    return data

def needs_normalization(data):
    """
    Determine if the given data needs normalization.

    Args:
        data (str or bytes): The data to be checked. Can be a string or bytes.

    Returns:
        str: "Yes" if the data needs normalization, "No" otherwise.
    """
    if isinstance(data, bytes):
        data = data.decode()
    # Check if the data matches the format that does not need normalization
    if re.search(r'\(\s*-?\d+(\.\d+)?(\+|\-)?\d*?j?\s*\)\s*\[.*?\]', data):
        return "No"
    else:
        return "Yes"

def normalize_data_format(data):
    """
    Normalize the format of the given data.

    Args:
        data (str or bytes): The data to be normalized. Can be a string or bytes.

    Returns:
        bytes: The normalized data as a byte string.
    """
    if isinstance(data, bytes):
        data = data.decode()
    normalized_data = []
    terms = data.split('+')
    for term in terms:
        term = term.strip()
        if term:
            match = re.match(r'(\S+)\s*\[(.*?)\]', term)
            if match:
                coeff = match.group(1)
                ops = match.group(2).strip()
                normalized_term = f"({coeff}) [{ops}]"
                normalized_data.append(normalized_term)
    return ' +\n'.join(normalized_data).encode()

def parse_hamiltonian_to_sparsepauliop(data):
    """
    Parse the Hamiltonian string into a list of SparsePauliOp terms.

    Args:
        data (str or bytes): The Hamiltonian data to be parsed. Can be a string or bytes.

    Returns:
        list: A list of tuples, where each tuple contains a dictionary representing the Pauli operators and 
              their corresponding qubit indices, and a complex coefficient.
    """
    if isinstance(data, bytes):
        data = data.decode()
    
    terms = re.findall(r'\(([^)]+)\)\s*\[(.*?)\]', data)
    
    parsed_pauli_list = []

    for coeff, ops in terms:
        coeff = complex(coeff.strip())
        ops_list = ops.split()
        pauli_dict = {}

        for op in ops_list:
            match = re.match(r'([XYZ])(\d+)', op)
            if match:
                pauli_op = match.group(1)
                qubit_index = int(match.group(2))
                pauli_dict[qubit_index] = pauli_op

        parsed_pauli_list.append((pauli_dict, coeff))
    
    return parsed_pauli_list

def determine_qubit_count(terms):
    """
    Determine the number of qubits required based on the given list of Pauli terms.

    Args:
        terms (list): A list of tuples, where each tuple contains a dictionary representing the Pauli operators and 
                      their corresponding qubit indices, and a complex coefficient.

    Returns:
        int: The number of qubits required.
    """
    max_qubit = 0
    for pauli_dict, _ in terms:
        if pauli_dict:
            max_in_term = max(pauli_dict.keys())
            if max_in_term > max_qubit:
                max_qubit = max_in_term
    return max_qubit + 1  # Since qubit indices start at 0

def download_and_extract(filename, url):
    """
    Download a file from a given URL and unzip it.

    Args:
        filename (str): The name of the file to be downloaded.
        url (str): The URL to download the file from.

    Returns:
        str: The path to the extracted file.
    """
    download_dir = "downloaded_hamlib_files"
    os.makedirs(download_dir, exist_ok=True)
    local_zip_path = os.path.join(download_dir, os.path.basename(url))
    
    # Download the file
    response = requests.get(url)
    if response.status_code == 200:
        with open(local_zip_path, 'wb') as file:
            file.write(response.content)
        # print(f"Downloaded {local_zip_path} successfully.")
    else:
        raise Exception(f"Failed to download from {url}.")

    # Unzip the file
    with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
        zip_ref.extractall(download_dir)
        # print(f"Extracted to {download_dir}.")
    
    # Return the path to the directory containing the extracted files
    return download_dir

def process_hamiltonian_file(filename, dataset_name):
    """
    Download the Hamiltonian file, extract it, and process the data to create a quantum circuit.

    Args:
        dataset_name (str): The name of the dataset to extract.
        filename (str): The name of the Hamiltonian file to be downloaded and processed.

    Returns:
        tuple: A tuple containing the constructed QuantumCircuit and the Hamiltonian as a SparsePauliOp.
    """

    ham_files = [h.file_name for h in hamiltonians]
    
    if filename in ham_files:
        url = hamiltonians[ham_files.index(filename)].url
        extracted_path = download_and_extract(filename, url)
        # Assuming the HDF5 file is located directly inside the extracted folder
        hdf5_file_path = os.path.join(extracted_path, filename)
        # print('hdf5_file_path', hdf5_file_path)
    else:
        raise ValueError(f"No URL mapping found for filename: {filename}")
    data = extract_dataset_hdf5(hdf5_file_path, dataset_name)
    # print(data)
    return data

def construct_dataset_name(file_key):
    """
    Construct a dataset name by reading specified properties from a JSON file.

    Args:
        file_key (str): The key corresponding to the dataset information in the JSON file.

    Returns:
        str: A constructed dataset name if successful, or an error message if not.

    Note:
        This function assumes the JSON file is named 'hamlib_parameter_use_input.json' and is located
        in the current working directory. The function reads the JSON file, retrieves properties 
        for the given file_key, and constructs a dataset name by concatenating these properties.
    """
    json_file_path = 'hamlib_parameter_use_input.json'

    # Try to open the JSON file and load data
    try:
        with open(json_file_path, 'r') as file:
            json_data = json.load(file)
    except FileNotFoundError:
        return "The specified JSON file could not be found."
    except json.JSONDecodeError:
        return "Error decoding JSON. Please check the file content."

    # Access the properties of the given file key from the JSON data
    file_properties = json_data.get(file_key)
    
    # Handle case where file_key is not found in data
    if not file_properties:
        return "File key not found in data"

    # Construct the dataset name dynamically
    dataset_parts = []
    for key, value in file_properties.items():
        dataset_parts.append(f"{key}-{value}")

    return '_'.join(dataset_parts)

def extract_variable_ranges(file_input):
    """
    Extracts the ranges of variable values from HDF5 files specified in the input.

    Args:
        file_input (list): A list of strings, each containing a function name, file path, 
                           and optionally a fixed variable with its value separated by colons.

    Returns:
        dict: A dictionary where keys are function names and values are dictionaries
              of variables and their possible values.
    """
    results = {}

    for entry in file_input:
        parts = entry.split(':')
        function_name, file_path = parts[0], parts[1]
        
        # Check if a fixed variable and value are provided
        if len(parts) > 2:
            fixed_var_value = parts[2]
            fixed_variable, fixed_value = fixed_var_value.split('=')
        else:
            fixed_variable, fixed_value = None, None

        # Dictionary to hold variables and their values
        variable_values = {}

        try:
            with h5py.File(file_path, 'r') as file:
                for item in file.keys():
                    # Assuming the format includes instance names in item or its attributes
                    instance_name = item.split(':')[0] if ':' in item else item
                    variables = parse_instance_variables(instance_name)

                    if fixed_variable is None or variables.get(fixed_variable) == fixed_value:
                        for var, val in variables.items():
                            if var not in variable_values:
                                variable_values[var] = set()
                            variable_values[var].add(val)
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

        # Store the results
        if variable_values:
            results[function_name] = variable_values
    
    # Print the results
    for function_name, variables in results.items():
        print(f"{function_name}:")
        for var, values in variables.items():
            # Use a sorting method that safely handles mixed data types
            sorted_values = sorted(values, key=lambda x: (is_numeric(x), float(x) if is_numeric(x) else x))
            print(f"  {var}: {sorted_values}")

def is_numeric(value):
    """ 
    Helper function to check if a string represents a numeric value. 
    """
    try:
        float(value)
        return True
    except ValueError:
        return False

def parse_instance_variables(instance_name):
    """
    Parses an instance name to extract variable names and their values.

    Args:
        instance_name (str): The name of the instance to be parsed.

    Returns:
        dict: A dictionary where keys are variable names and values are their corresponding values.
    """
    parts = instance_name.split('_')
    variables = {}
    for part in parts:
        if '-' in part:
            index = part.find('-')
            var = part[:index]
            val = part[index+1:]
            variables[var] = val
    return variables

def generate_json_for_hdf5_input(file_input):
    """
    Generate a JSON file from a list of HDF5 file inputs describing various Hamiltonians.

    Args:
        file_input (list of str): A list containing entries of the format 'function_name:file_path[:fixed_var=value]'.
            Each entry represents an HDF5 file along with optional fixed variable information.

    """
    results = {}

    for entry in file_input:
        parts = entry.split(':')
        function_name, file_path = parts[0], parts[1]
        base_filename = os.path.basename(file_path)

        # Check if file exists, if not handle it accordingly
        if not os.path.exists(file_path):
            process_hamiltonian_file(base_filename, "")  # Your custom handling for missing files
            continue  # Skip processing if file is not found or after handling

        if len(parts) > 2:
            fixed_var_value = parts[2]
            fixed_variable, fixed_value = fixed_var_value.split('=')
        else:
            fixed_variable, fixed_value = None, None

        variable_values = {}

        try:
            with h5py.File(file_path, 'r') as file:
                for item in file.keys():
                    instance_name = item.split(':')[0] if ':' in item else item
                    variables = parse_instance_variables(instance_name)

                    if fixed_variable is None or variables.get(fixed_variable) == fixed_value:
                        for var, val in variables.items():
                            # Store only the first encountered value for each variable
                            if var not in variable_values:
                                variable_values[var] = val
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

        if variable_values:
            results[base_filename] = variable_values

    with open('downloaded_hamlib_files/sample_input_json.json', 'w') as f:
        json.dump(results, f, indent=2)


def view_hdf5_structure():
    """
    A sample function to view the structure of specific HDF5 files and their variable ranges.
    """
    verbose = False
    file_input = [
        "tfim1:downloaded_hamlib_files/tfim.hdf5:graph=1D-grid-pbc-qubitnodes",
        "tfim2:downloaded_hamlib_files/tfim.hdf5",
        "fermi-hubbard:downloaded_hamlib_files/FH_D-1.hdf5",
        "max3sat:downloaded_hamlib_files/random_max3sat-hams.hdf5",
        "heis:downloaded_hamlib_files/heis.hdf5",
        "bh:downloaded_hamlib_files/BH_D-1_d-4.hdf5"
        # Add more entries as needed
    ]
    for entry in file_input:
        parts = entry.split(':')
        filename = parts[1]  # Extract the full path of the file
        base_filename = os.path.basename(filename)  # Extract the base filename

        if not os.path.exists(filename):
            process_hamiltonian_file(base_filename, "")
    extract_variable_ranges(file_input)
    generate_json_for_hdf5_input(file_input)


#######################
# MAIN

# DEVNOTE: Need to modify the args below to specify options for printing info about the HamLib file content

import argparse
def get_args():
    parser = argparse.ArgumentParser(description="Bernstei-Vazirani Benchmark")
    #parser.add_argument("--api", "-a", default=None, help="Programming API", type=str)
    #parser.add_argument("--target", "-t", default=None, help="Target Backend", type=str)
    parser.add_argument("--backend_id", "-b", default=None, help="Backend Identifier", type=str)
    parser.add_argument("--num_shots", "-s", default=100, help="Number of shots", type=int)
    parser.add_argument("--num_qubits", "-n", default=0, help="Number of qubits (min = max = N)", type=int)
    parser.add_argument("--min_qubits", "-min", default=3, help="Minimum number of qubits", type=int)
    parser.add_argument("--max_qubits", "-max", default=8, help="Maximum number of qubits", type=int)
    parser.add_argument("--skip_qubits", "-k", default=1, help="Number of qubits to skip", type=int)
    parser.add_argument("--max_circuits", "-c", default=3, help="Maximum circuit repetitions", type=int)     
    parser.add_argument("--hamiltonian", "-ham", default="heisenberg", help="Name of Hamiltonian", type=str)
    parser.add_argument("--method", "-m", default=1, help="Algorithm Method", type=int)
    parser.add_argument("--use_XX_YY_ZZ_gates", action="store_true", help="Use explicit XX, YY, ZZ gates")
    #parser.add_argument("--theta", default=0.0, help="Input Theta Value", type=float)
    parser.add_argument("--nonoise", "-non", action="store_true", help="Use Noiseless Simulator")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose")
    return parser.parse_args()
 
# if main, execute method
if __name__ == '__main__':   
    args = get_args()
    
    # configure the QED-C Benchmark package for use with the given API
    # (done here so we can set verbose for now)
    #PhaseEstimation, kernel_draw = qedc_benchmarks_init(args.api)
    
    # special argument handling
    #ex.verbose = args.verbose
    #verbose = args.verbose
    
    if args.num_qubits > 0: args.min_qubits = args.max_qubits = args.num_qubits
    
    print("\n\n\n\nPrinting the structure of the hdf5 file")
    view_hdf5_structure()
    '''
    # execute benchmark program
    run(min_qubits=args.min_qubits, max_qubits=args.max_qubits,
        skip_qubits=args.skip_qubits, max_circuits=args.max_circuits,
        num_shots=args.num_shots,
        hamiltonian=args.hamiltonian,
        method=args.method,
        use_XX_YY_ZZ_gates = args.use_XX_YY_ZZ_gates,
        #theta=args.theta,
        backend_id=args.backend_id,
        exec_options = {"noise_model" : None} if args.nonoise else {},
        #api=args.api
        )
    '''

