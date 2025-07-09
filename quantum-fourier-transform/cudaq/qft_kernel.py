'''
Quantum Fourier Transform Benchmark Program - CUDA Quantum Kernel
(C) Quantum Economic Development Consortium (QED-C) 2024.
'''

# DEVNOTE: Method 2 of this benchmark does not work correctly due to limitations in the ability
# for the Python version of cudaq to collect and return an array of measured values (Issue #????).
# Only the final measurements are returned, meaning the fidelity is not determined correctly.
import cudaq

from typing import List

# saved circuits for display
QC_ = None
Uf_ = None

############### QFT Circuit Definition

# Inverse Quantum Fourier Transform
@cudaq.kernel
def iqft(register: cudaq.qview):
	M_PI = 3.1415926536
	
	input_size = register.size()
			
	# Generate multiple groups of diminishing angle CRZs and H gate
	for i_qubit in range(input_size):
		ri_qubit = input_size - i_qubit - 1				# map to cudaq qubits
		
		# precede with an H gate (applied to all qubits)
		h(register[ri_qubit])
		
		# number of controlled Z rotations to perform at this level
		num_crzs = input_size - i_qubit - 1
		
		# if not the highest order qubit, add multiple controlled RZs of decreasing angle
		if i_qubit < input_size - 1:   
			for j in range(0, num_crzs):
				divisor = 2 ** (j + 1)
				r1.ctrl( -M_PI / divisor , register[ri_qubit], register[ri_qubit - j - 1])		
				
# Quantum Fourier Transform
@cudaq.kernel
def qft(register: cudaq.qview):
	M_PI = 3.1415926536
	
	input_size = register.size()

	# Generate multiple groups of diminishing angle CRZs and H gate
	for i_qubit in range(input_size):
		ri_qubit = input_size - i_qubit - 1			# map to cudaq qubits
		
		# number of controlled Z rotations to perform at this level
		num_crzs = i_qubit
		
		# if not the highest order qubit, add multiple controlled RZs of decreasing angle
		#if i_qubit > 0:   
		if i_qubit <= input_size - 1: 
			for j in range(0, num_crzs):
				rj = num_crzs - j - 1
				divisor = 2 ** (rj + 1)
				r1.ctrl( M_PI / divisor , register[i_qubit], register[i_qubit - rj - 1])
				
		# follow each set of rotations with an H gate (applied to all qubits)
		h(register[i_qubit])

@cudaq.kernel			
def qft_kernel (num_qubits: int, secret_int: int, init_phases: List[float], method: int = 1):
	M_PI = 3.1415926536
	
	# Allocate the specified number of qubits - this
	# corresponds to the length of the hidden bitstring.
	qubits = cudaq.qvector(num_qubits)
	
	# method 1 is the mirror circuit version of QFT followed by IQFT
	if method == 1:

		# Rotate each qubit into its initial state, 0 or 1
		for index, phase in enumerate(init_phases):
			if phase > 0:
				x(qubits[num_qubits - index - 1])	
		
		barrier(qubits, num_qubits)
			
		# Apply inverse quantum Fourier transform
		qft(qubits)

		barrier(qubits, num_qubits)
			
		# some compilers recognize the QFT and IQFT in series and collapse them to identity;
		# perform a set of rotations to add one to the secret_int to avoid this collapse
		for i_q in range(0, num_qubits):
			ri_q = num_qubits - i_q - 1
			divisor = 2 ** (i_q)
			rz( 1 * M_PI / divisor , qubits[ri_q])
		
		barrier(qubits, num_qubits)
			
		# Apply inverse quantum Fourier transform
		iqft(qubits)

		# Measure to gather sampling statistics
		mz(qubits)
		
	# method 2 is just the IQFT
	elif method == 2:
	
		# Measure to gather sampling statistics
		mz(qubits)
		
	pass

#DEVNOTE: use this as a barrier when drawing circuit; comment out otherwise
@cudaq.kernel
def barrier(qubits: cudaq.qview, num_qubits: int):
	for i in range(num_qubits / 2):
		swap(qubits[i*2], qubits[i*2 + 1])
		swap(qubits[i*2], qubits[i*2 + 1])
			
			
def QuantumFourierTransform (num_qubits: int, secret_int: int, init_phase: List[float], method: int = 1, use_midcircuit_measurement: bool = False):

	qc = [qft_kernel, [num_qubits, secret_int, init_phase, method]]
	
	global QC_
	if num_qubits <= 6:
		QC_ = qc

	return qc

############### BV Circuit Drawer

# Draw the circuits of this benchmark program
def kernel_draw():
	print("Sample Circuit:");
	if QC_ != None:
		print(cudaq.draw(QC_[0], *QC_[1]))
	else:
		print("	 ... too large!")
	
	 