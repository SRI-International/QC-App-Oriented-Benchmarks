import openfermion as of
import h5py
from hamlib_utils import (
    hamiltonians,
    download_and_extract
)


def parse_through_hdf5(func):
    """
    Decorator function that iterates through an HDF5 file and performs
    the action specified by â€˜ func â€˜ on the internal and leaf nodes in the
    HDF5 file.
    """

    def wrapper(obj, path='/', key=None):
        if type(obj) in [h5py._hl.group.Group, h5py._hl.files.File]:
            for ky in obj.keys():
                func(obj, path, key=ky, leaf=False)
                wrapper(obj=obj[ky], path=path + ky + ',', key=ky)
        elif type(obj) is h5py._hl.dataset.Dataset:
            func(obj, path, key=None, leaf=True)
    return wrapper


def print_hdf5_structure(fname_hdf5: str):
    """
    Print the path structure of the HDF5 file.

    Args
    ----
    fname_hdf5 ( str ) : full path where HDF5 file is stored
    """

    @parse_through_hdf5
    def action(obj, path='/', key=None, leaf=False):
        if key is not None:
            print(
                (path.count('/') - 1) * '\t', '-', key, ':', path + key + '/'
            )
        if leaf:
            print((path.count('/') - 1) * '\t', '[^^ DATASET ^^]')

    with h5py.File(fname_hdf5, 'r') as f:
        action(f['/'])


def get_hdf5_keys(fname_hdf5: str):
    """ Get a list of keys to all datasets stored in the HDF5 file.

    Args
    ----
    fname_hdf5 ( str ) : full path where HDF5 file is stored
    """

    all_keys = []

    @parse_through_hdf5
    def action(obj, path='/', key=None, leaf=False):
        if leaf is True:
            all_keys.append(path)

    with h5py.File(fname_hdf5, 'r') as f:
        action(f['/'])
    return all_keys


def read_openfermion_hdf5(fname_hdf5: str, key: str, optype=of.QubitOperator):
    """
    Read any openfermion operator object from HDF5 file at specified key.
    'optype' is the op class, can be of.QubitOperator or of.FermionOperator.
    """

    with h5py.File(fname_hdf5, 'r', libver='latest') as f:
        op = optype(f[key][()].decode("utf-8"))
    return op


if __name__ == "__main__":
    for ham in hamiltonians[0:1]:  # change to iterate over all the data
        path = download_and_extract(ham.file_name, ham.url)
        the_file = f'{path}/{ham.file_name}'
        #print_hdf5_structure(the_file) # uncomment for optional printing
        keys = get_hdf5_keys(the_file)
        for key in keys:
            if "1D" in key:
                op = read_openfermion_hdf5(the_file, key[:-1]) # :-1 to remove the trailing ,
                print(f'{key} : {of.count_qubits(op)}')
