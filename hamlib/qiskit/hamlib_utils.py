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

def sparse_pauliop(terms, num_qubits):
    """
    Construct a SparsePauliOp from a list of Pauli terms and the number of qubits.

    Args:
        terms (list): A list of tuples, where each tuple contains a dictionary representing the Pauli operators and 
                      their corresponding qubit indices, and a complex coefficient.
        num_qubits (int): The total number of qubits.

    Returns:
        SparsePauliOp: The Hamiltonian represented as a SparsePauliOp.
    """
    pauli_list = []
    
    for pauli_dict, coefficient in terms:
        label = ['I'] * num_qubits  # Start with identity on all qubits
        for qubit, pauli_op in pauli_dict.items():
            label[qubit] = pauli_op
        label = ''.join(label)
        pauli_list.append((label, coefficient))
    
    hamiltonian = SparsePauliOp.from_list(pauli_list, num_qubits=num_qubits)
    return hamiltonian



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
        print(f"Downloaded {local_zip_path} successfully.")
    else:
        raise Exception(f"Failed to download from {url}.")

    # Unzip the file
    with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
        zip_ref.extractall(download_dir)
        print(f"Extracted to {download_dir}.")
    
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
    url_mapping = {
        'tfim.hdf5': 'https://portal.nersc.gov/cfs/m888/dcamps/hamlib/condensedmatter/tfim/tfim.zip',
        'FH_D-1.hdf5': 'https://portal.nersc.gov/cfs/m888/dcamps/hamlib/condensedmatter/fermihubbard/FH_D-1.zip'
        # Add more mappings as needed
    }
    
    if filename in url_mapping:
        url = url_mapping[filename]
        extracted_path = download_and_extract(filename, url)
        print('downloaded_path',extracted_path)
        # Assuming the HDF5 file is located directly inside the extracted folder
        hdf5_file_path = os.path.join(extracted_path, filename)
        print('hdf5_file_path', hdf5_file_path)
    else:
        raise ValueError(f"No URL mapping found for filename: {filename}")
    data = extract_dataset_hdf5(hdf5_file_path, dataset_name)
    print(data)
    return data



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
    
    # dataset_name = 'graph-1D-grid-nonpbc-qubitnodes_Lx-4_h-0.1'
    # filename = 'tfim.hdf5'
    dataset_name = 'fh-graph-1D-grid-nonpbc-qubitnodes_Lx-3_U-0_enc-jw'
    filename = 'FH_D-1.hdf5'
    
    data = process_hamiltonian_file(filename, dataset_name)
    if data is not None:
        print("Raw Hamiltonian Data: ",data)
    else:
        print("No data extracted.")
        
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

