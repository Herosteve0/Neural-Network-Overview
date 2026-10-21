using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using NeuralNetwork.Core.System;
using NeuralNetwork.Core.System.Backend;
using NeuralNetwork.Core.System.Data;
using GpuBackend = NeuralNetwork.Core.System.Backend.GpuBackend;
using SimdBackend = NeuralNetwork.Core.System.Backend.SimdBackend;

/*
 * Benchmarks:
 * 
 * Cpu: Intel i5 14400K, 16 logical processors, 4GHz
 * Gpu: NVIDIA GeForce RTX 3060
 * 
 * Trainer:
 * - Batch Size: 2500
 * - Breath Ticks: 5000
 * - Learning Rate: 0.085f, 0.9f, 5
 *
 * Tester:
 * - Batch Size: 2500
 * 
 * Batch Recycles: 1
 * Random: 1000
 *
 * Scalar:              I am not testing that xd
 * Scalar MultiThread:  01:24:12.0713327
 *
 * Simd:                00:12:23.7781757
 * Simd MultiThread:    00:02:20.3753676
 *
 * Gpu:                 00:01:36.5960609
 * Gpu MultiThread:     00:01:53.8817631
 * 
 */

// NetworkManager.DefaultInitialize();
// await NetworkManager.GroupTraining();

DataBatch<float> trainingData = new DataBatch<float>(MnistDatabase.LoadAllTrainingData());
DataBatch<float> testingData = new DataBatch<float>(MnistDatabase.LoadAllTestingData());

Context context = Context.Create(builder => builder.Cuda().EnableAlgorithms());
Accelerator accelerator = context.Devices[0].CreateAccelerator(context);

const bool multiThread = false;

Trainer trainer = new Trainer(2500, 5000, new LearningRateManager(0.085f, 0.9f, 5), multiThread);
trainer.LoadBatch(trainingData, 25);

Tester tester = new Tester(2500, multiThread);
tester.LoadBatch(testingData);

Random random = new Random(1000);

NetworkFile storer = new NetworkFile("save.nn");
NetworkInstance network = storer.Load(new GpuBackend(accelerator));
// NetworkInstance network = storer.Load(new SimdBackend());
// NetworkInstance network = storer.Load(new ScalarBackend());

float oldAcc = network.Accuracy;
if (oldAcc is < 0 or > 1) {
    Console.WriteLine("File accuracy found corrupted. Setting it to 0");
    oldAcc = 0;
} else {
    Console.WriteLine($"Old Accuracy: {oldAcc * 100f}%");
}

for (int i = 0; i < 5; i++) {
    float learningRate = LearningRateManager.SetFromAccuracy(oldAcc);
    Console.WriteLine($"Training Loop {i + 1} with LearningRate: {learningRate}");
    TimeSpan time = await trainer.Train(network, random, learningRate);
    Console.WriteLine($"Time: {time}, Learning Rate: {trainer.Tracker.LearningRate}, Accuracy: {trainer.Tracker.Accuracy}, Loss: {trainer.Tracker.Loss}");

    float accuracy = await tester.Test(network);
    Console.WriteLine($"Accuracy: {accuracy * 100f}%");

    if (oldAcc < accuracy) {
        storer.Save(new NetworkInstance(network, accuracy));
        oldAcc = accuracy;
    }
}

network.Dispose();

namespace NeuralNetwork.Core {
    internal class AdvancedRunner(Accelerator accelerator, int processors, int[] layers, DataBatch<float> trainingData, int recycles, DataBatch<float> testingData) {
        private readonly NetworkGrower _grower = new NetworkGrower(processors, trainingData, recycles, testingData);
    
        public async Task RunSave(int generations, NetworkGrowerInfo info, bool save = false) {
            IBackend backend = new GpuBackend(accelerator);
            NetworkFile storer = new NetworkFile("save.nn");
        
            GrowableNetwork network = new GrowableNetwork(storer.Load(backend));
            await ExecuteRun(network, generations, info, save);
    
            network.Dispose();
        }

        public async Task RunNew(int generations, NetworkGrowerInfo info, bool save = false) {
            IBackend backend = new GpuBackend(accelerator);
            Random random = new Random(info.RngSeed);

            GrowableNetwork network = new GrowableNetwork(layers, backend, random);

            await ExecuteRun(network, generations, info, save);

            network.Dispose();
        }

        private async Task ExecuteRun(GrowableNetwork network, int generations, NetworkGrowerInfo info, bool save) {
            int oldAccuracy = (int)(network.Accuracy * testingData.Size);
        
            GrowableNetwork newNetwork = await _grower.GenTrain(
                network,
                generations,
                info
            );
        
            int newAccuracy = (int)(newNetwork.Accuracy * testingData.Size);

            int deltaAccuracy = newAccuracy - oldAccuracy;
            Console.WriteLine($"Accuracy changed by {deltaAccuracy}");
            if (save && deltaAccuracy > 0) {
                Console.WriteLine($"Neural Network saved.");

                NetworkFile storer = new NetworkFile("save.nn");
                storer.Save(newNetwork);
            }
        }
    }

    internal class SimpleRunner(Accelerator accelerator, int[] layers, DataBatch<float> trainingData, DataBatch<float> testingData) {
        public async Task RunSave(bool save = false) {
            IBackend backend = new SimdBackend();
            Random random = new Random(50);

            // Get the previous network
            NetworkFile storer = new NetworkFile("save.nn");
            NetworkInstance instance = storer.Load(backend);

            await ExecuteRun(instance, random, save);

            instance.Dispose();
        }

        public async Task RunNew(bool save = false) {
            IBackend backend = new GpuBackend(accelerator);
            // IBackend backend =  new SimdBackend();
            Random random = new Random(50);

            Network network = new Network(layers, backend, random);
            NetworkInstance instance = new NetworkInstance(network, 0);

            await ExecuteRun(instance, random, save);
            Tester.DrawWeights(network);

            instance.Dispose();
        }

        private async Task ExecuteRun(NetworkInstance instance, Random random, bool save) {
            int oldAccuracy = (int)(instance.Accuracy * testingData.Size);
            float accuracy = await TrainAndTest(instance.Network, random);

            NetworkInstance newInstance = instance with { Accuracy = accuracy };

            float deltaAccuracy = accuracy - oldAccuracy;
            Console.WriteLine($"Accuracy changed by {deltaAccuracy * 100f}%");
            if (save && deltaAccuracy > 0) {
                Console.WriteLine($"Neural Network saved.");

                NetworkFile storer = new NetworkFile("save.nn");
                storer.Save(newInstance);
            }
        }

        private async Task<float> TrainAndTest(Network network, Random random) {
            Trainer trainer = new Trainer(5000, 5000, new LearningRateManager(0.085f, 0.9f, 3));
            Tester tester = new Tester(10000);

            trainer.LoadBatch(trainingData);
            tester.LoadBatch(testingData);
            
            await trainer.Train(network, random);
            float accuracy = await tester.Test(network);

            return accuracy;
        }
    }
}