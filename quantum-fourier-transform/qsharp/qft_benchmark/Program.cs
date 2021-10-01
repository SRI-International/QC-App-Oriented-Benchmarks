using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;
using System.Threading.Tasks;
using Microsoft.Quantum.Simulation.Common;
using Microsoft.Quantum.Simulation.Core;
using Microsoft.Quantum.Simulation.Simulators;
using Microsoft.Quantum.Simulation.Simulators.QCTraceSimulators;

namespace qft_benchmark
{
    class Program
    {
        public static void Main(string[] args)
        {
            Console.WriteLine("test");
            RunTest();
        }


        private static void RunTest(int MinSampleRatePower = 3, int MaxSampleRatePower = 8, int MaxQubits = 9, int Shots = 1)
        {
            // Validate parameters (smallest circuit is 3 qubits)
            MaxSampleRatePower = Math.Max(3, MaxSampleRatePower);
            MinSampleRatePower = Math.Min(Math.Max(3, MinSampleRatePower), MaxSampleRatePower);

            int minFrequency = 1;
            int maxFrequency = 3;

            Stopwatch timer = new Stopwatch();
            using (QuantumSimulator simulator = new())
            {
                for (int sampleRatePower = MinSampleRatePower; sampleRatePower <= MaxSampleRatePower; sampleRatePower++)
                {
                    for (int frequency = minFrequency; frequency <= maxFrequency; frequency++)
                    {
                        for (int numberOfQubits = sampleRatePower; numberOfQubits <= MaxQubits; numberOfQubits++)
                        {
                            for (int i = 0; i < Shots; i++)
                            {
                                timer.Restart();
                                Sine_Test.Run(simulator, sampleRatePower, frequency, numberOfQubits, true).Wait();
                                timer.Stop();
                                Console.WriteLine($"Ran tests with {numberOfQubits} qubits in {timer.Elapsed}.");

                                ResourcesEstimator estimator = new(new QCTraceSimulatorConfiguration
                                {
                                    UseDepthCounter = true,
                                    UseWidthCounter = true,
                                    UsePrimitiveOperationsCounter = true,
                                    ThrowOnUnconstrainedMeasurement = false
                                });
                                Sine_Test.Run(estimator, sampleRatePower, frequency, numberOfQubits, false).Wait();
                                Console.WriteLine(estimator.ToTSV());
                                Console.WriteLine();
                            }
                        }
                    }
                }

                timer.Restart();
                Period_6_Test.Run(simulator, true).Wait();
                timer.Stop();
                Console.WriteLine($"Ran period 6 test in {timer.Elapsed}.");
                
                ResourcesEstimator estimator2 = new(new QCTraceSimulatorConfiguration
                {
                    UseDepthCounter = true,
                    UseWidthCounter = true,
                    UsePrimitiveOperationsCounter = true,
                    ThrowOnUnconstrainedMeasurement = false
                });
                Period_6_Test.Run(estimator2, false).Wait();
                Console.WriteLine(estimator2.ToTSV());
                Console.WriteLine();
            }

        }

    }
}
