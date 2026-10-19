using ILGPU.Runtime;
using NeuralNetwork.Core;
using NeuralNetwork.Core.Data;

NetworkManager.DefaultInitialize();

await NetworkManager.GroupTraining();

// TestProgram();

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

        private async Task<int> TrainAndTest(Network network, Random random) {
            Trainer trainer = new Trainer(5000, 5000, new LearningRateManager(0.085f, 0.9f, 3));
            Tester tester = new Tester(10000);

            trainer.LoadBatch(trainingData);
            tester.LoadBatch(testingData);
            
            await trainer.Train(network, random);
            int accuracy = await tester.Test(network);

            return accuracy;
        }
    }