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

namespace qft_benchmark
{
    open Microsoft.Quantum.Diagnostics;
    open Microsoft.Quantum.Intrinsic;
    open Microsoft.Quantum.Canon;
    open Microsoft.Quantum.Preparation;
    open Microsoft.Quantum.Math;
    open Microsoft.Quantum.Arithmetic;
    open Microsoft.Quantum.Convert;
	open Qedc.Qft;

	/// # Summary
	/// Tests my QFT implementation by comparing it to the classical DFT, ensuring it produces the
	/// same output as DFT when given the same input (after being normalized for quantum operations).
	/// 
	/// # Input
	/// ## PrepOperation
	/// The operation that prepares the qubit register in the desired state for this test.
	/// 
	/// ## NumberOfQubits
	/// The size of the processing register to use, in qubits. This will be used to represent 2^N
	/// samples of the input signal.
	/// 
	/// ## SampleRate
	/// The sampling rate used by the prep opration. This is used to determine the actual frequency
	/// of the measured value once QFT is finished, which can vary based on the number of samples
	/// and the sample rate.
	/// 
	/// ## CorrectFrequency
	/// The correct answer that QFT should provide after running on the prepared input state.
	/// 
	/// ## Validate
	/// True to validate if the QFT worked correctly during simulation, false to ignore it for resource
	/// estimation.
	operation TestQftWithWaveformSamples(
		PrepOperation : (Qubit[] => Unit),
		NumberOfQubits : Int,
		SampleRate : Double,
		CorrectFrequency : Double,
		Validate : Bool
	) : Unit
	{
		use register = Qubit[NumberOfQubits]
		{
			// Set up the register so it's in the correct state for the test
			PrepOperation(register);

			// Run the inverse QFT, which corresponds to the normal DFT
			Adjoint Qft(register);

			if(Validate)
			{
				// Measure the result from QFT
				let numberOfStates = IntAsDouble(2 ^ NumberOfQubits);
				mutable result = IntAsDouble(MeasureInteger(BigEndianAsLittleEndian(BigEndian(register))));

				// QFT suffers from the same Nyquist-frequency mirroring as DFT, but we can't just
				// look at all of the output details and ignore the mirrored results. If we end up
				// measuring a mirrored result, this will flip it back to the proper result in the
				// 0 < X < N/2 space.
				if(result > numberOfStates / 2.0)
				{
					set result = numberOfStates - result;
				}

				// Correct for the sample rate.
				let totalTime = numberOfStates / SampleRate;
				set result = result / totalTime;

				// Verify we got the right result, and clean up
				if(result != CorrectFrequency)
				{
					fail $"Expected frequency {CorrectFrequency} but measured {result}.";
				}
			}
			ResetAll(register);
		}
	}

	/// # Summary
	/// Prepares a qubit register so the amplitudes of each state correspond to the values
	/// of the sine or cosine function at the timestep represented by that state. For example:
	/// with a 1 Hz sine wave, 8 samples per second and 8 total samples, the sine wave samples
	/// will be [0, 0.707, 1, 0.707, 0, -0.707, -1, -0.707]. This function will put the register
	/// into the corresponding quantum state: 0.354*|001⟩ + 0.5*|010⟩ + 0.354*|011⟩ - 0.354*|101⟩
	/// - 0.5*|110⟩ - 0.354*|111⟩. Note that the amplitudes will be normalized so the state is a
	/// unit vector.
	/// 
	/// # Input
	/// ## Frequency
	/// The frequency of the wave, in Hz.
	/// 
	/// ## SampleRate
	/// The number of samples to take per second.
	/// 
	/// ## Register
	/// The qubit register to encode the sine wave samples into.
	/// 
	/// ## UseCosine
	/// True to use the cosine function, false to use the sine function.
	operation PrepareSineWaveSamples(
		Frequency : Double,
		SampleRate : Double,
		Register : Qubit[],
		UseCosine : Bool
	) : Unit
	{
		let numberOfSamples = 2 ^ Length(Register);
		// Since quantum states need to be unit vectors, this will be used to reduce the
		// sin / cos output properly.
		let normalizationFactor = Sqrt(IntAsDouble(numberOfSamples) / 2.0);
		
		mutable samples = new ComplexPolar[numberOfSamples];
		for i in 0..numberOfSamples - 1
		{
			let timestamp = IntAsDouble(i) / SampleRate;
			mutable sample = 0.0;
			if(UseCosine)
			{
				set sample = Cos(Frequency * 2.0 * PI() * timestamp) / normalizationFactor;
			}
			else
			{
				set sample = Sin(Frequency * 2.0 * PI() * timestamp) / normalizationFactor;
			}
			set samples w/= i <- ComplexPolar(sample, 0.0);
		}

		// This is such a handy function. Props to the guys that wrote it.
		PrepareArbitraryStateCP(samples, BigEndianAsLittleEndian(BigEndian(Register)));
	}

	/// # Summary
	/// Tests QFT by generating a sine wave with the specified properties, encoding it into the amplitudes
	/// of a superposition, running IQFT on it, and ensuring that the correct frequency was recovered.
	/// 
	/// # Input
	/// ## SampleRatePower
	/// The power of 2 that describes the number of samples the wave should represent.
	/// 
	/// ## Frequency
	/// The frequency of the sine way to encode into the qubits during the test.
	/// 
	/// ## NumberOfQubits
	/// The number of qubits to use in the sample register
	/// 
	/// ## Validate
	/// True to validate if the QFT worked correctly during simulation, false to ignore it for resource
	/// estimation.
	operation Sine_Test(SampleRatePower : Int, Frequency : Int, NumberOfQubits : Int, Validate : Bool) : Unit
	{
		let sampleRate = 2 ^ SampleRatePower;
		let sampleRateAsDouble = IntAsDouble(sampleRate);
		let frequencyAsDouble = IntAsDouble(Frequency);

		Message($"Running sine test for {Frequency} Hz, {NumberOfQubits} qubits " + 
			$"({2^NumberOfQubits} samples), {sampleRate} samples per second.");
		mutable prepFunction = PrepareSineWaveSamples(frequencyAsDouble, sampleRateAsDouble, _, false);
		TestQftWithWaveformSamples(prepFunction, NumberOfQubits, sampleRateAsDouble, frequencyAsDouble, Validate);
		Message($"Passed!");
		Message("");

		//Message($"Running cosine test for {Frequency} Hz, {NumberOfQubits} qubits " + 
		//	$"({2^NumberOfQubits} samples), {sampleRate} samples per second.");
		//set prepFunction = PrepareSineWaveSamples(frequencyAsDouble, sampleRateAsDouble, _, true);
		//TestQftWithWaveformSamples(prepFunction, NumberOfQubits, sampleRateAsDouble, frequencyAsDouble, Validate);
		//Message($"Passed!");
		//Message("");
	}

	/// # Summary
	/// Tests QFT by running a single iteration of the period-finding subroutine from
	/// Shor's algorithm. This test will use 21 as the number to factor, 11 as the
	/// original guess, and ensure that QFT reports that the modular exponential
	/// equation has a period of 6.
	operation Period_6_Test(Validate : Bool) : Unit
	{
		// So this test basically just runs a hardcoded iteration of the quantum portion
		// of Shor's algorithm. I don't want to explain the entire thing here; you can
		// look at Shor.qs for my implementation, which has plenty of documentation
		// attached to it. For this test, I'm trying to factor 21. That means the
		// "output" register needs to be 5 qubits (because 2^4 = 16 and 2^5 = 32, so it
		// needs 5 qubits to be represented in binary). For the input register, I'm going
		// with 9 qubits: 21^2 = 441, 2^8 = 256, and 2^9 = 512, so 21^2 needs 9 qubits to
		// be represented in binary. That will give 512 discrete states. For a guess of
		// 11, the period will be 6:
		// -------------------------
		//  State (i) | 11^i mod 21
		// -------------------------
		//          0 | 1
		//          1 | 11
		//          2 | 16
		//          3 | 8
		//          4 | 4
		//          5 | 2
		//          6 | 1	<== Pattern repeats here, after 6 entries
		//          7 | 11
		//          8 | 16
		//          ...
		//
		// QFT should return some value X which, when divided by 512, should be really
		// close to 0/6, 1/6, 2/6, 3/6, 4/6, or 5/6. The amplitude peaks (the expected
		// values) are 0, 85, 171, 256, 341, and 427.

		let inputLength = 9;
		let outputLength = 5;
		let numberToFactor = 21;
		let guess = 11;
		use (input, output) = (Qubit[inputLength], Qubit[outputLength])
		{
            Message("Running a single iteration of the Shor subroutine on 11^x mod 21 (period 6)");

			ApplyToEach(H, input);		// Input = |+...+>
			X(output[outputLength-1]);	// Output = |0...01>
			let outputAsLE = LittleEndian(output);

			// Do the arithmetic so the input register is entangled with the output register; after
			// this, if the state X is measured on the input register, the output register will always
			// be measured as 11^X mod 21.
			for i in inputLength - 1..-1..0
			{
				let powerOfTwo = 1 <<< (inputLength - 1 - i);

				let constant = ExpModI(guess, powerOfTwo, numberToFactor);
				Controlled MultiplyByModularInteger([input[i]],
					(constant, numberToFactor, outputAsLE));
			}
			
			// Run inverse QFT (the analog of the normal DFT) to find the period
			Adjoint Qft(input);

			if(Validate)
			{
				// Measure the resulting period and make sure it's close to a multiple of 1/6,
				// with a tolerance of 0.01.
				let measurement = MeasureInteger(BigEndianAsLittleEndian(BigEndian(input)));
				let scaledMeasurement = IntAsDouble(measurement) / 512.0 * 6.0;
				let nearestMultiple = Round(scaledMeasurement);
				let delta = AbsD(scaledMeasurement - IntAsDouble(nearestMultiple));

				Message($"Measured {measurement}/512 => {scaledMeasurement}, delta = {delta}");
				EqualityFactB(delta < 0.05, true, $"QFT failed, delta of {delta} is too high.");
			}

			ResetAll(input);
			ResetAll(output);
		}
	}

	/// # Summary
	/// Takes an operation that takes in a BigEndian as an argument
	/// and a qubit array, and runs the operation with the qubit
	/// array as the argument.
	/// 
	/// # Input
	/// ## Operation
	/// The operation to run the input register with
	/// 
	/// ## Register
	/// The qubit array to run the operation on
	/// 
	/// # Remarks
	/// This is a helper function which is necessary to run the
	/// reference implementation of QFT included in the Q# libraries
	/// with the AssertOperationsEqualReferenced function, used to
	/// compare my implementation to it.
	operation ConvertArrayToBigEndian(
		Operation : (BigEndian => Unit is Adj),
		Register : Qubit[]
	) : Unit
	{
		body(...)
		{
			// This literally just wraps the register in a BigEndian
			// and runs the function on it.
			let bigEndianRegister = BigEndian(Register);
			Operation(bigEndianRegister);
		}
		adjoint invert;
	}

	/// # Summary
	/// Compares my implementation of QFT to Q#'s reference implementation.
	/// Note that this assumes the canonical implementation is correct,
	/// because it doesn't actually calculate the QFT of anything and
	/// verify for correctness - it just makes sure my version does the same
	/// thing as their version.
	operation ReferenceComparison_Test() : Unit
	{
		for i in 1..10
		{
			Message($"Running QFT reference tests with {i} qubits...");
			let referenceQftWithArrays = ConvertArrayToBigEndian(QFT, _);
			AssertOperationsEqualReferenced(i, Qft, referenceQftWithArrays);
			Message("Passed!");
			Message("");
		}
	}
	
}