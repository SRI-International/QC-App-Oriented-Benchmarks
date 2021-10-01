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


// This file contains the implementation and tests for the Deutsch-Jozsa algorithm.
// It checks a given function f(x) that takes in a register and outputs a single bit,
// to see whether or not it's "constant" or "balanced". Constant means it always
// returns 0 (or always 1) for all possible inputs. Balanced means it returns 0 for
// half of the possible inputs and 1 for the other half.
// Normally this would take N/2 + 1 checks (N = the number of possible inputs) in order
// to prove with 100% certainty that the algorithm was one or the other, but the DJ
// algorithm takes advantage of the way the Hadamard gate works to cause quantum
// interference, which can produce the answer in only 1 check.
// 
// This is a "toy" algorithm in the sense that it was invented by mathematicians as
// the first simple example of a problem that offers a speedup with quantum computers
// compared to classical computers, but it's a super contrived problem with no real
// practical applications.
namespace Qedc.DeutschJozsa
{
    open Microsoft.Quantum.Diagnostics;
    open Microsoft.Quantum.Intrinsic;
    open Microsoft.Quantum.Canon;
	open QSharpOracles;
	

	// ==============================
	// == Algorithm Implementation ==
	// ==============================


	/// # Summary
	/// Runs the Deutsch-Jozsa algorithm on the provided oracle, determining
	/// whether it's constant or balanced.
	/// 
	/// # Input
	/// ## Register
	/// The register of qubits to use for running the algorithm. These should
	/// be in the |0...0> state. The register can be any length you want.
	/// 
	/// ## Oracle
	/// The oracle function containing the implementation of the function you're
	/// trying to inspect. It should be a standard bit flipping oracle, which
	/// will flip the target qubit if and only if the input register meets some
	/// particular criteria.
	/// 
	/// # Output
	/// Returns false if the provided oracle is a balanced function, or true
	/// if it's a constant function.
    operation DeutschJozsa(
		Register : Qubit[],
		Oracle : ((Qubit[], Qubit) => Unit is Adj)
	) : Bool
	{
		// Initialize the qubits to |+...+>
		ApplyToEach(H, Register);

		// Run the oracle in phase-flip mode. Any of the superposition states that
        // triggered the oracle will have their phase flipped. The only way to get
        // back to the |0...0> state with a mass-Hadamard operation is if all of
        // the phases are the same, which corresponds to a constant function. A
        // balanced function will only flip half of them, which will put the register
        // into some other state that's not |0...0> after a mass-Hadamard.
		use target = Qubit()
		{
			X(target);
			Oracle(Register, target);
			X(target);
		}
		
		// Bring the register back to the computational basis and measure each
		// qubit. If it's |0...0>, we know it's constant. If it's literally anything
		// else, it's balanced.
		for i in 0..Length(Register) - 1
		{
			H(Register[i]);
			if(M(Register[i]) == One)
			{
				return false;
			}
		}

		return true;
    }

}