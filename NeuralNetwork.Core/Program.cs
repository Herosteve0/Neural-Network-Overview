using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using NeuralNetwork.Core.Backend;
using NeuralNetwork.Core.Data;
using NeuralNetwork.Core.Iterators;
using NeuralNetwork.Core.NetworkFormats;
using NeuralNetwork.Core.System;

using GpuBackend = NeuralNetwork.Core.Backend.GpuBackend;
using SimdBackend = NeuralNetwork.Core.Backend.SimdBackend;

/*
 * Benchmarks:
 * 
 * Cpu: Intel i5 14400K, 16 logical processors, 4GHz
 * Gpu: NVIDIA GeForce RTX 3060
 *
 * Layers: 784, 512, 512, 512, 512, 10
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
 * Layers: 784, 8192, 8192, 8192, 10
 * 
 * Gpu:                 00:01:36.5960609
 * Gpu MultiThread:     00:01:53.8817631
 */

DataBatch<float> trainingData = new DataBatch<float>(MnistDatabase.LoadAllTrainingData());
DataBatch<float> testingData = new DataBatch<float>(MnistDatabase.LoadAllTestingData());

Context context = Context.Create(builder => builder.Cuda().EnableAlgorithms());
Accelerator accelerator = context.Devices[0].CreateAccelerator(context);

// new ProgramTester(accelerator).StartTesting();

await SimpleExecution("save.nn", 5, 20, false);

// await ParallelExecution("save.nn", 4, 1, 1,
//     new NetworkGrowerInfo(
//         1000,
//         new LearningRateManager(0.085f, 0.9f, 5),
//         new MutationStrengthManager(0.025f, 0.75f)
//     )
// );
return;

async Task ParallelExecution(string saveFile, int processors, int generations, int recycles, NetworkGrowerInfo info) {
    NetworkGrower grower = new NetworkGrower(processors,
        new LearningRateManager(0.085f, 0.9f, 5)
    );
    grower.LoadBatch(trainingData, recycles, testingData);
    
    NetworkFile storer = new NetworkFile(saveFile);
    GrowableNetwork network = storer.Load(new GpuBackend(accelerator));
    float oldAcc = GetAccuracy(network);

    Console.WriteLine($"GenTraining on {generations} generations.");
    network = await grower.GenTrain(
        network,
        generations,
        info
    );

    float accuracy = network.Accuracy;
    Console.WriteLine($"Finished with accuracy: {accuracy * 100f}%");
    
    if (oldAcc < accuracy) {
        storer.Save(network);
    }
    
    network.Dispose();
}

async Task SimpleExecution(string saveFile, int iterations, int recycles, bool multiThread) {
    Trainer trainer = new Trainer(2500, 1, 5000, new LearningRateManager(0.085f, 0.9f, 5), multiThread);
    trainer.LoadBatch(trainingData, recycles);

    Tester tester = new Tester(2500, multiThread);
    tester.LoadBatch(testingData);

    Random random = new Random(1000);

    NetworkFile storer = new NetworkFile(saveFile);

    IBackend backend = new GpuBackend(accelerator);
    // IBackend backend = new SimdBackend();
    // IBackend backend = new ScalarBackend();

    // NetworkInstance network = storer.Load(backend) ?? new NetworkInstance(new Network([784, 16, 16, 10], backend, random), 0f);
    NetworkInstance network = storer.Load(backend) ?? new NetworkInstance(new Network([784, 2048, 2048, 10], backend, random), 0f);
    Console.WriteLine($"Network Layers: " + string.Join(", ", network.Network.LayerLength));
    
    float oldAcc = GetAccuracy(network);

    for (int i = 0; i < iterations; i++) {
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
}

float GetAccuracy(NetworkInstance instance) {
    float oldAcc = instance.Accuracy;
    if (oldAcc is < 0 or > 1) {
        Console.WriteLine("File accuracy found corrupted. Setting it to 0");
        return 0f;
    }
    Console.WriteLine($"Old Accuracy: {oldAcc * 100f}%");
    return oldAcc;
}