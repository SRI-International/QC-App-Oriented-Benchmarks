using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;
using System.Threading.Tasks;
using Microsoft.Quantum.Simulation.Common;
using Microsoft.Quantum.Simulation.Core;
using Microsoft.Quantum.Simulation.Simulators;
using Microsoft.Quantum.Simulation.Simulators.QCTraceSimulators;

namespace deutsch_jozsa
{
    class Program
    {
        public static void Main(string[] args)
        {
            Console.WriteLine("test");
            RunTest();
        }


        private static void RunTest(int MinQubits = 3, int MaxQubits = 8, int Shots = 1)
        {
            // Validate parameters (smallest circuit is 3 qubits)
            MaxQubits = Math.Max(3, MaxQubits);
            MinQubits = Math.Min(Math.Max(3, MinQubits), MaxQubits);

            Stopwatch timer = new Stopwatch();
            Func<IOperationFactory, long, bool, Task<bool>>[] tests =
            {
                ConstantZero_Test.Run,
                ConstantOne_Test.Run,
                OddNumberOfOnes_Test.Run,
                NthQubitParity_Test.Run
            };
            using (QuantumSimulator simulator = new())
            {
                for (int numberOfQubits = MinQubits; numberOfQubits <= MaxQubits; numberOfQubits++)
                {
                    foreach (Func<IOperationFactory, long, bool, Task<bool>> test in tests)
                    {
                        for (int i = 0; i < Shots; i++)
                        {
                            timer.Restart();
                            test(simulator, numberOfQubits, true).Wait();
                            timer.Stop();
                            Console.WriteLine($"Ran tests with {numberOfQubits} qubits in {timer.Elapsed}.");

                            ResourcesEstimator estimator = new(new QCTraceSimulatorConfiguration
                            {
                                UseDepthCounter = true,
                                UseWidthCounter = true,
                                ThrowOnUnconstrainedMeasurement = false
                            });
                            test(estimator, numberOfQubits, false).Wait();
                            Console.WriteLine(estimator.ToTSV());
                            Console.WriteLine();
                        }
                    }
                }
            }

        }

    }
}
