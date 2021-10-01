// ========================================================================
// Copyright (C) 2021 The MITRE Corporation.
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
//     http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ========================================================================


// This file contains some tests for the Quantum Fourier Transform
// (QFT) algorithm. 
// 
// Note that Q# provides a canonical QFT implementation as part of
// their own libraries, seen here:
// https://docs.microsoft.com/en-us/qsharp/api/canon/microsoft.quantum.canon.qft
// The source for it is here:
// https://github.com/Microsoft/QuantumLibraries/blob/master/Canon/src/QFT.qs
namespace Qedc.Qft
{
    open Microsoft.Quantum.Diagnostics;
    open Microsoft.Quantum.Preparation;
    open Microsoft.Quantum.Math;
    open Microsoft.Quantum.Arithmetic;
    open Microsoft.Quantum.Convert;
    open Microsoft.Quantum.Intrinsic;
    open Microsoft.Quantum.Canon;


	// ==============================
	// == Algorithm Implementation ==
	// ==============================

    
	/// # Summary
	/// Performs an in-place quantum fourier transform on the given register.
	/// 
	/// # Input
	/// ## Register
	/// The array of qubits representing the data to be transformed. This will
	/// be treated as a big endian integer, where the first qubit represents the
	/// most significant bit.
	/// 
	/// # Remarks
	/// Note that by the established conventions, the QFT corresponds to the
	/// inverse classical DFT, and the adjoint QFT corresponds to the normal
	/// DFT.
	operation Qft(Register : Qubit[]) : Unit
	{
		body(...)
		{
			for i in 0..Length(Register) - 1
			{
				// Each qubit starts with a Hadamard
				H(Register[i]);

				// Go through the rest of the qubits that follow this one,
				// we're going to use them as control qubits on phase-shift
				// gates. The phase-shift gate is basically a gate that rotates
				// the |1> portion of a qubit's state around the Z axis of the
				// Bloch sphere by Φ, where Φ is the angle from the +X axis on
				// the X-Y plane. Q# actually provides this gate as the R1
				// function, and for convenience when Φ = kπ/2^m for some
				// numbers k and m, they provide the R1Frac function which just
				// lets you specify k and m directly.
				// 
				// For more info on the phase-shift gate, look at the "phase shift"
				// section of this Wiki article and the MSDN page for R1Frac:
				// https://en.wikipedia.org/wiki/Quantum_logic_gate
				// https://docs.microsoft.com/en-us/qsharp/api/prelude/microsoft.quantum.primitive.r1frac
				for j in i + 1..Length(Register) - 1
				{
					// According to the circuit diagram, the controlled R1 gates
					// change the "m" value as described above. The first one
					// is always 2, and then it iterates from there until the
					// last one.
					let m = j - i + 1;

					// Perform the rotation, controlled by the jth qubit on the
					// ith qubit, with e^(2πi/2^m)
					Controlled R1Frac([Register[j]], (2, m, Register[i]));
				}
			}

			// The bit order is going to be backwards after the QFT so this just
			// reverses it.
			SwapReverseRegister(Register);
		}

		adjoint invert;
	}

}