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

namespace deutsch_jozsa
{
    open Microsoft.Quantum.Diagnostics;
    open Microsoft.Quantum.Intrinsic;
    open Microsoft.Quantum.Canon;
	open QSharpOracles;
	open Qedc.DeutschJozsa;

    

	/// # Summary
	/// Runs the Deutsch-Jozsa algorithm on the provided oracle, ensuring
	/// that it correctly identifies the oracle as constant or balanced.
	/// 
	/// ## OracleName
	/// A simple description of what this oracle checks for.
	/// 
	/// ## Oracle
	/// The oracle function containing the implementation of the function you're
	/// trying to inspect. It should be a standard bit flipping oracle, which
	/// will flip the target qubit if and only if the input register meets some
	/// particular criteria.
	/// 
	/// ## ShouldBeConstant
	/// True if the oracle is a constant one (always returns 0 or always 
	/// returns 1 no matter what the input is), false if it's a balanced one
	/// (returns 0 for half the input, returns 1 for the other half).
	/// 
	/// ## Validate
	/// True to validate the output (check if the oracle and DJ algorithm worked
	/// correctly) for simulation, false to ignore it for resource estimation.
	/// 
	/// ## NumberOfQubits
	/// The number of qubits to use in the input register when evaluating the
	/// oracle. This should be an even number to ensure that the oracle could
	/// be truly balanced.
	/// 
	/// # Output
	/// Returns true if the oracle was measured to be constant, false if it was
	/// measured to be balanced.
	operation RunTest(
		OracleName : String,
		Oracle : ((Qubit[], Qubit) => Unit is Adj),
		ShouldBeConstant : Bool,
		NumberOfQubits : Int,
		Validate : Bool
	) : Bool
	{
		Message($"Running DJ with the {OracleName} oracle on {NumberOfQubits} qubits.");

		// Run the algorithm and make sure it gives the right answer
		mutable result = false;
		use qubits = Qubit[NumberOfQubits]
		{
			set result = DeutschJozsa(qubits, Oracle);

			if(Validate)
			{
				let constantString = "constant";
				let balancedString = "balanced";
				EqualityFactB(result, ShouldBeConstant,
					$"Test failed: {OracleName} should be " + 
					$"{ShouldBeConstant ? constantString | balancedString}" +
					$"but the algorithm says it was " +
					$"{ShouldBeConstant ? balancedString | constantString}.");
			}

			ResetAll(qubits); // Lazy qubit cleanup
		}

		Message($"Passed!");
		return result;
	}
	
	/// # Summary
	/// Runs the test on the constant zero function.
	operation ConstantZero_Test(NumberOfQubits : Int, Validate: Bool) : Bool
	{
		return RunTest("constant zero", AlwaysZero, true, NumberOfQubits, Validate);
	}
	
	/// # Summary
	/// Runs the test on the constant one function.
	operation ConstantOne_Test(NumberOfQubits : Int, Validate: Bool) : Bool
	{
		return RunTest("constant one", AlwaysOne, true, NumberOfQubits, Validate);
	}
	
	/// # Summary
	/// Runs the test on the odd number of |1> state check.
	operation OddNumberOfOnes_Test(NumberOfQubits : Int, Validate: Bool) : Bool
	{
		return RunTest("odd number of |1> check", CheckForOddNumberOfOnes, false, NumberOfQubits, Validate);
	}
	
	/// # Summary
	/// Runs the test on the Nth-qubit parity check function.
	operation NthQubitParity_Test(NumberOfQubits : Int, Validate: Bool) : Bool
	{
		// Run it N times by iterating through the index to check, just to be thorough
		mutable result = false;
		for i in 0..NumberOfQubits - 1
		{
			set result = RunTest(
				$"q{i} parity check",
				CheckIfQubitIsOne(_, i, _), // Partial function with the index already in place
				false,
				NumberOfQubits, 
				Validate);
		}
		return result;
	}
	
}