'''
Quantum Fourier Transform Benchmark Program - CUDA Quantum Kernel
(C) Quantum Economic Development Consortium (QED-C) 2024.
'''

import cudaq
import math

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
def qft_kernel(num_qubits: int, secret_int: int, init_phases: List[float],
               method: int = 1, use_midcircuit_measurement: bool = False):
	M_PI = 3.1415926536
	qubits = cudaq.qvector(num_qubits)

	if method == 1:
		for index, phase in enumerate(init_phases):
			if phase > 0:
				x(qubits[num_qubits - index - 1])

		qft(qubits)

		for i_q in range(0, num_qubits):
			ri_q = num_qubits - i_q - 1
			divisor = 2 ** i_q
			rz(M_PI / divisor, qubits[ri_q])

		iqft(qubits)
		mz(qubits)

	elif method == 2:
		for i_q in range(num_qubits):
			h(qubits[i_q])

		for i_q in range(num_qubits):
			ri_q = num_qubits - i_q - 1
			rz(init_phases[i_q], qubits[ri_q])

		iqft(qubits)
		mz(qubits)

	elif method == 3:
		safe_secret_int = secret_int
		if safe_secret_int > num_qubits:
			safe_secret_int = num_qubits

		for i_q in range(safe_secret_int):
			h(qubits[num_qubits - i_q - 1])

		for i_q in range(safe_secret_int, num_qubits):
			x(qubits[num_qubits - i_q - 1])

		iqft(qubits)
		mz(qubits)


@cudaq.kernel
def qft_midcircuit_kernel(num_qubits: int, secret_int: int,
                          init_phases: List[float], method: int = 1) -> int:
	M_PI = 3.1415926536
	qubits = cudaq.qvector(num_qubits)

	if method == 1:
		for index, phase in enumerate(init_phases):
			if phase > 0:
				x(qubits[num_qubits - index - 1])

		qft(qubits)

		for i_q in range(0, num_qubits):
			ri_q = num_qubits - i_q - 1
			divisor = 2 ** i_q
			rz(M_PI / divisor, qubits[ri_q])

	elif method == 2:
		for i_q in range(num_qubits):
			h(qubits[i_q])

		for i_q in range(num_qubits):
			ri_q = num_qubits - i_q - 1
			rz(init_phases[i_q], qubits[ri_q])

	elif method == 3:
		safe_secret_int = secret_int
		if safe_secret_int > num_qubits:
			safe_secret_int = num_qubits

		for i_q in range(safe_secret_int):
			h(qubits[num_qubits - i_q - 1])

		for i_q in range(safe_secret_int, num_qubits):
			x(qubits[num_qubits - i_q - 1])

	data = 0
	input_size = qubits.size()
	for i_qubit in range(input_size):
		ri_qubit = input_size - i_qubit - 1
		h(qubits[ri_qubit])

		meas = mz(qubits[ri_qubit])
		if meas:
			data = data + (1 << i_qubit)
			if i_qubit < input_size - 1:
				num_crzs = input_size - i_qubit - 1
				for j in range(0, num_crzs):
					divisor = 2 ** (j + 1)
					rz(-M_PI / divisor, qubits[ri_qubit - j - 1])

	return data


#DEVNOTE: use this as a barrier when drawing circuit; comment out otherwise
@cudaq.kernel
def barrier(qubits: cudaq.qview, num_qubits: int):
	for i in range(num_qubits / 2):
		swap(qubits[i*2], qubits[i*2 + 1])
		swap(qubits[i*2], qubits[i*2 + 1])
			
			
def QuantumFourierTransform (num_qubits: int, secret_int: int, init_phase: List[float], method: int = 1, use_midcircuit_measurement: bool = False):

	if method == 2:
		init_phase = [
			(secret_int % (2 ** (i_q + 1))) * math.pi / (2 ** i_q)
			for i_q in range(num_qubits)
		]

	if use_midcircuit_measurement:
		qc = [qft_midcircuit_kernel, [num_qubits, secret_int, init_phase, method],
		      {"result_width": num_qubits}]
	elif method == 3:
		qc = [qft_kernel, [num_qubits, secret_int, init_phase, method, use_midcircuit_measurement],
		      {"counts_dict": True}]
	else:
		qc = [qft_kernel, [num_qubits, secret_int, init_phase, method, use_midcircuit_measurement]]

	global QC_
	if num_qubits <= 6:
		QC_ = qc

	return qc

############### BV Circuit Drawer

# Draw the circuits of this benchmark program
def kernel_draw():
	print("Sample Circuit:");
	if QC_ != None:
		try:
			print(cudaq.draw(QC_[0], *QC_[1]))
		except Exception as ex:
			print(f"ERROR attempting to draw the kernel")
			print(ex)
	else:
		print("	 ... too large!")
	
	 


	 
