using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using Neurtal_Network_Overview.System;
using Neurtal_Network_Overview.System.Data;

Context context = Context.Create(builder => builder
    .Cuda()
    // .OpenCL()
    .EnableAlgorithms()
    );

Console.WriteLine("Available devices:");
foreach (Device device in context) {
    Console.WriteLine($"- {device.Name}");
}
Accelerator accelerator = context.GetPreferredDevice(preferCPU: false).CreateAccelerator(context);


DataBatch<float> trainingData = new DataBatch<float>(MnistDatabase.LoadAllTrainingData());
DataBatch<float> testingData = new DataBatch<float>(MnistDatabase.LoadAllTestingData());

// GpuMultiTrainer multiTrainer = new GpuMultiTrainer(1, accelerator, [784, 8128, 8128, 10], trainingData, testingData);
// NetworkInstance instance = await multiTrainer.TrainProcess(5000, 1, 1);

await SimpleNewRun();
return;

async Task SimpleSavedRun(bool save = false) {
    IBackend backend = new SimdBackend();
    Random random = new Random(50);
    
    // Get the previous network
    NetworkFile storer = new NetworkFile("save.nn");
    NetworkInstance instance = storer.Load(backend);
    
    await ExecuteRun(instance, random, save);
    
    instance.Dispose();
}

async Task SimpleNewRun(bool save = false) {
    IBackend backend =  new GpuBackend(accelerator);
    // IBackend backend =  new SimdBackend();
    Random random = new Random(50);
    
    NeuralNetwork network =  new NeuralNetwork([784, 10], backend, random);
    NetworkInstance instance = new NetworkInstance(network, 0);
    
    await ExecuteRun(instance, random, save);
    Tester.DrawWeights(network);
    
    instance.Dispose();
}

async Task ExecuteRun(NetworkInstance instance, Random random, bool save) {
    int oldAccuracy = instance.Accuracy;
    int accuracy = await TrainAndTest(instance.Network, random);

    NetworkInstance newInstance = instance with { Accuracy = accuracy };
    
    int deltaAccuracy = accuracy - oldAccuracy;
    Console.WriteLine($"Accuracy changed by {deltaAccuracy}");
    if (save && deltaAccuracy > 0) {
        Console.WriteLine($"Neural Network saved.");

        NetworkFile storer = new NetworkFile("save.nn");
        storer.Save(newInstance);
    }
}

async Task<int> TrainAndTest(NeuralNetwork network, Random random) {
    Trainer trainer =  new Trainer(network, 5000, 5000, new TrainerLearningInfo(0.0075f));
    Tester tester =   new Tester(network, 10000);
    
    await trainer.Train(trainingData, 1, random);
    int accuracy = await tester.Test(testingData, false);

    return accuracy;
}

void TestProgram() {
    ProgramTester programTester = new ProgramTester(accelerator);
    
    programTester.StartTesting();
}

public abstract class ProgramExecutor {
    public static bool MultiThread;
    
    public ProgramExecutor(bool multiThread) {
        MultiThread = multiThread;
    }
}