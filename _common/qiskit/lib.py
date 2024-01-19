# Return expected measurement array scaled to number of shots executed
def get_expectation(num_qubits, degree, num_shots):

    # find expectation counts for the given circuit 
    id = f"_{num_qubits}_{degree}"
    if id in expectations:
        counts = expectations[id]
        
        # scale to number of shots
        for k, v in counts.items():
            counts[k] = round(v * num_shots)
        
        # delete from the dictionary
        del expectations[id]
        
        return counts
        
    else:
        return None
