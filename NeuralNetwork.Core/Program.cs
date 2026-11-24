using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using NeuralNetwork.Core.Backend;
using NeuralNetwork.Core.Data;
using NeuralNetwork.Core.Iterators;
using NeuralNetwork.Core.NetworkFormats;
using NeuralNetwork.Core.NetworkFormats.Layers;
using NeuralNetwork.Core.Others;

/*
 * Benchmarks: (Outdated)
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
 * Quick laptop benchmarks
 *
 * Cpu: AMD Ryzen 9 4900H, 16 logical processors, 4GHz
 * Gpu: NVIDIA GeForce GTX 1660 Ti 6GB
 * 
 * Layers: 784, 64, 64, 10
 *
 * Scalar (Single Thread):  00:06:44.2514893
 * Simd (Single Thread):    00:00:30.0934393
 * Gpu (Single Thread):     00:00:29.3940633
 *
 * Scalar (Multi Thread):   00:01:55.2662609
 * Simd (Multi Thread):     00:00:04.5962730
 * Gpu (Multi Thread):      00:01:11.9218895
 */

DataBatch<float> trainingData = new DataBatch<float>(MnistDatabase.LoadAllTrainingData());
DataBatch<float> testingData = new DataBatch<float>(MnistDatabase.LoadAllTestingData());

Context context = Context.Create(builder => builder.Cuda().EnableAlgorithms());
Accelerator accelerator = context.Devices[0].CreateAccelerator(context);

// await TestRun();
await SimpleExecution("save.nn", new GpuBackend(accelerator), 1, 1, false);
await SimpleExecution("save.nn", new GpuBackend(accelerator), 20, 4, false);
return;

async Task SimpleExecution(string saveFile, IBackend backend, int iterations, int recycles, bool multiThread) {
    Trainer trainer = new Trainer(2500, 1, 5000, new LearningRate(0.085f, 0.9f, 5), multiThread);
    trainer.LoadBatch(trainingData, recycles);

    Tester tester = new Tester(2500, multiThread);
    tester.LoadBatch(testingData);

    Random random = new Random(1000);

    NetworkFile storer = new NetworkFile(saveFile);
    
    ILoss lossFunction = new CrossEntropy();

    // NetworkInstance network = storer.Load(backend) ?? new NetworkInstance(new Network([784, 16, 16, 10], backend, random), 0f);
    NetworkInstance network = storer.Load(backend, lossFunction) ?? new NetworkInstance(
        new Network(
            backend,
            lossFunction,
            random,
            
            new InputLayer(784),
            new FullyConnectedLayer(512, new ReLu()),
            new DropoutLayer(0.2f, 100),
            new FullyConnectedLayer(512, new ReLu()),
            new DropoutLayer(0.2f, 200),
            new FullyConnectedLayer(512, new ReLu()),
            new DropoutLayer(0.2f, 300),
            new FullyConnectedLayer(512, new ReLu()),
            new FullyConnectedLayer(10, new Softmax())
        ), 0f
    );

    Console.WriteLine("Network Layers:");
    foreach (BaseLayer layer in network.Network.Layers) {
        Console.WriteLine($"- {layer.GetType().Name}: {layer.NeuronCount}");
    }
    Console.WriteLine();
    
    float oldAcc = GetAccuracy(network);

    for (int i = 0; i < iterations; i++) {
        float learningRate = LearningRate.SetFromAccuracy(oldAcc);
        Console.WriteLine($"Training Loop {i + 1}/{iterations} with LearningRate: {learningRate}");
        
        TimeSpan time = await trainer.Train(network, random, learningRate);
        Console.WriteLine($"- Training Time:        {time}\n" +
                          $"- Final Learning Rate:  {trainer.Tracker.LearningRate}\n" +
                          $"- Training Accuracy:    {trainer.Tracker.Accuracy}\n" +
                          $"- Training Loss:        {trainer.Tracker.Loss}");

        float accuracy = await tester.Test(network);
        Console.WriteLine($"- Accuracy: {accuracy * 100f}%");

        if (oldAcc < accuracy) {
            storer.Save(new NetworkInstance(network, accuracy));
            oldAcc = accuracy;
        }
    }

    network.Dispose();
}

async Task TestRun() {
    IBackend[] backends = [new ScalarBackend(), new SimdBackend(), new GpuBackend(accelerator)];
    ILoss lossFunction = new CrossEntropy();
    
    Console.WriteLine();

    foreach (bool multithread in new[] { false, true }) {
        Trainer trainer = new Trainer(2500, 1, 5000, new LearningRate(0.085f, 0.9f, 5), multithread);
        trainer.LoadBatch(trainingData);

        Tester tester = new Tester(2500, multithread);
        tester.LoadBatch(testingData);
    
        foreach (IBackend backend in backends) {
            Random random = new Random(1000);
            Network network = new Network(backend, lossFunction, random,
                new InputLayer(784),
                new FullyConnectedLayer(64, new ReLu()),
                new FullyConnectedLayer(64, new ReLu()),
                new FullyConnectedLayer(10, new Softmax())
            );

            await trainer.Train(network, random);
            float accuracy = await tester.Test(network);

            Console.WriteLine($"[{trainer.Tracker.TimeSinceStart + tester.Tracker.TimeSinceStart}] " + (multithread ? "Multithreaded " : "") + $"{backend.GetType().Name} Accuracy: {accuracy * 100f}%");

            network.Dispose();
        }
    }
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