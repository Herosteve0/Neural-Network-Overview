using System.Numerics;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using ILGPU.Runtime.OpenCL;
using NeuralNetwork.Core.System.Backend;
using NeuralNetwork.Core.System.Data;

namespace NeuralNetwork.Core.System;

public static class NetworkManager {
    private static readonly DataBatch<float> TrainingData = new DataBatch<float>(MnistDatabase.LoadAllTrainingData());
    private static readonly DataBatch<float> TestingData = new DataBatch<float>(MnistDatabase.LoadAllTestingData());

    private static int[] _layers;
    private static int _rngSeed;
    private static int _processors;
    private static int _generations;
    
    private static float _defaultLearningRate;
    private static int _repeats;
    private static float _mutationStrength;
    private static float _mutationDecay;

    private static NetworkFile _storer;
    
    private static Accelerator _accelerator;
    private static IBackend _backend;

    public static void DefaultInitialize() {
        // Please ensure you are initializing the correct hardware
        InitializeAccelerator(0);
        InitializeBackend();
        InitializeStorer("save.nn");
        InitializeParameters(
            [784, 512, 512, 512, 512, 10],
            5000,
            2,
            1,
            0.085f,
            1,
            0.025f,
            0.75f
        );
        LoadNetwork();
    }
    
    // Programmer Runners

    public static NetworkGrower Grower;
    public static GrowableNetwork Network;
    
    public static async Task GroupTraining(bool save = false) {
        if (_accelerator == null) throw new NullReferenceException("No accelerator loaded.");
        
        Grower = new NetworkGrower(_processors, TrainingData, _repeats, TestingData);
        Network = await Grower.GenTrain(
            Network,
            _generations,
            new NetworkGrowerInfo(
                _defaultLearningRate,
                _rngSeed,
                new MutationStrengthManager(
                    _mutationStrength,
                    _mutationDecay
                )
            )
        );
        
        Console.WriteLine($"Accuracy: {Network.Accuracy}%");
        
        SaveNetwork();
    }

    public static void ForceStop() {
        Grower?.ForceStop();
    }
    
    // Storing

    public static void InitializeStorer(string filepath) {
        _storer = new NetworkFile(filepath);
    }

    public static void NewNetwork() {
        Network?.Dispose();
        Network = new GrowableNetwork(_layers, _backend, new Random(_rngSeed));
    }

    public static void LoadNetwork() {
        Network?.Dispose();
        Network = new GrowableNetwork(_storer.Load(_backend));
    }

    public static void SaveNetwork() {
        if (Network == null) return;
        if (_storer.LoadAccuracy() < Network.Accuracy) {
            _storer.Save(Network);
        }
    }
    public static void ForceSaveNetwork() {
        if (Network == null) return;
        _storer.Save(Network);
    }
    
    // Parameters

    public static void InitializeParameters(int[] layers, int rngSeed, int generations, int processors,
        float defaultLearningRate, int repeats, float mutationStrength, float mutationDecay
    ) {
        _layers = layers;
        _rngSeed = rngSeed;

        _processors = processors;
        _generations = generations;
        
        _defaultLearningRate = defaultLearningRate;
        _repeats = repeats;
        _mutationStrength = mutationStrength;
        _mutationDecay = mutationDecay;
    }
    
    // Hardware

    public static void InitializeHardware(
        int deviceIndex = 0, bool cuda = true, bool openCl = true, bool useCpu = true
    ) {
        InitializeAccelerator(deviceIndex, cuda, openCl, useCpu);
        InitializeBackend();
    }
    
    public static void InitializeAccelerator(
        int deviceIndex = 0, bool cuda = true, bool openCl = true, bool useCpu = true
    ) {
        _accelerator = GenerateAccelerator(deviceIndex, cuda, openCl, useCpu);
    }
    
    public static void InitializeBackend() {
        _backend =  GenerateGpuBackend(_accelerator);
        if (_backend != null) return;

        _backend = GenerateSimdBackend();
        if (_backend != null) return;

        _backend = GenerateScalarBackend();
        if (_backend != null) return;
        
        Console.WriteLine("Failed to generate backend.");
    }

    private static GpuBackend GenerateGpuBackend(Accelerator accelerator) {
        if (accelerator == null) return null;
        return new GpuBackend(accelerator);
    }
    private static Accelerator GenerateAccelerator(int deviceIndex, bool cuda, bool openCl, bool useCpu) {
        Context context = Context.Create(builder => {
            try {
                if (cuda) builder.Cuda();
            } catch (Exception ex) {
                Console.WriteLine(ex);
            }
            
            try {
                if (openCl) builder.OpenCL();
            } catch (Exception ex) {
                Console.WriteLine(ex);
            }
            
            builder.EnableAlgorithms();
        });
        
        Console.WriteLine("Available devices:");
        int i = 0;
        foreach (Device device in context) {
            Console.WriteLine(
                deviceIndex == i ? $"  >>: {device.Name}" : $"  {i}: {device.Name}"
            );
            i++;
        }

        if (i == 0) return null;
        
        return context.Devices[deviceIndex].CreateAccelerator(context);
    }

    private static SimdBackend GenerateSimdBackend() {
        if (!Vector.IsHardwareAccelerated) return null;
        return new SimdBackend();
    }

    private static ScalarBackend GenerateScalarBackend() {
        return new ScalarBackend();
    }
}