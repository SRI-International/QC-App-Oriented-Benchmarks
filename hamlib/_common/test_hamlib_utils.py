
from hamlib._common import hamlib_utils

hamlib_utils.verbose = True

# this should give error
hamlib_utils.load_hamlib_file("abc")

# this should give error
hamlib_utils.find_dataset_for_params(12, None)

####################

hamlib_utils.load_hamlib_file("FH_D-1")

hamlib_utils.find_dataset_for_params(12, { "graph-1D-grid": "pbc", "enc": "bk", "U":4 })

####################

hamlib_utils.load_hamlib_file("tfim")

# this should return an array of 9 datasets
hamlib_utils.find_dataset_for_params(12, { "graph-1D-grid": "pbc" })

hamlib_utils.find_dataset_for_params(12, { "graph-1D-grid": "pbc", "h": 0.5 })

####################

hamlib_utils.load_hamlib_file("H2")

# this is an example of a search with a parameter with no value
hamlib_utils.find_dataset_for_params(12, { "ham_JW": "" } )

####################

hamlib_utils.load_hamlib_file("chemistry/vibrational/all-vib-o3")

# this is another example of a search with a parameter with no value
# Unfortunately it finds two datasets; need to find way to force it to be unique
# this is a limitation for now
hamlib_utils.find_dataset_for_params(12, { "enc_unary": "" } )
