using ILGPU.Runtime;
using NeuralNetwork.Core.System.Backend;
using NeuralNetwork.Core.System.Data;

namespace NeuralNetwork.Core.System;

using ForwardData = (float[][] values, float[][] activations);
using ForwardNoise = (float[] values, float[] activations);

using BackwardData = (float[][] delta, float[][] deltaWeight, float[][] deltaBias);
using BackwardNoise = (float[] delta, float[] deltaWeight, float[] deltaBias);

public class ProgramTester(Accelerator accelerator) {
    private readonly TestNetwork _scalar = new TestNetwork(new ScalarBackend());
    private readonly TestNetwork _simd = new TestNetwork(new SimdBackend());
    private readonly TestNetwork _gpu = new TestNetwork(new GpuBackend(accelerator));
    
    private readonly DataSample<float> _dataSample = MnistDatabase.LoadAllTrainingData()[0];

    public void StartTesting() {
        TestLoad();
        TestInitialization();
        TestForward();
        TestPrediction();
        TestLoss();
        TestBackward();
        TestAdjust();
    }

    private void TestLoad() {
        Console.WriteLine("Loading data sample.");
        
        _scalar.TestLoad(_dataSample);
        _simd.TestLoad(_dataSample);
        _gpu.TestLoad(_dataSample);
        
        Console.WriteLine("");
    }

    private void TestInitialization() {
        Console.WriteLine("Testing Initialization.");
        
        _scalar.TestNetworkInitialization();
        _simd.TestNetworkInitialization();
        _gpu.TestNetworkInitialization();
        
        Console.WriteLine("");
    }
    
    private void TestForward() {
        _scalar.IterateForward();
        _simd.IterateForward();
        _gpu.IterateForward();
        
        CompareForwardLayers();
        
        Console.WriteLine("");
    }

    private void TestPrediction() {
        Console.WriteLine("Testing prediction.");
        
        _scalar.TestPrediction();
        _simd.TestPrediction();
        _gpu.TestPrediction();
        
        Console.WriteLine("");
    }

    private void TestLoss() {
        Console.WriteLine("Testing loss.");
        
        _scalar.TestLoss();
        _simd.TestLoss();
        _gpu.TestLoss();
        
        Console.WriteLine("");
    }
    
    private void TestBackward() {
        _scalar.IterateBackward();
        _simd.IterateBackward();
        _gpu.IterateBackward();
        
        CompareBackwardLayers();
        
        Console.WriteLine("");
    }
    
    private void TestAdjust() {
        _scalar.IterateAdjust();
        _simd.IterateAdjust();
        _gpu.IterateAdjust();
        
        CompareForwardLayers();
        
        Console.WriteLine("");
    }

    private void CompareForwardLayers() {
        Console.WriteLine("Comparing layers.");

        ForwardData scalarContext = _scalar.ReadForwardLayers();
        ForwardData simdContext = _simd.ReadForwardLayers();
        ForwardData gpuContext = _gpu.ReadForwardLayers();
        
        Console.WriteLine("Scalar parameters:");
        Console.WriteLine("- Values:");
        for (int i = 0; i < scalarContext.values.Length; i++) {
            float min = scalarContext.values[i].Min();
            float max = scalarContext.values[i].Max();
            float avg = scalarContext.values[i].Average();
            Console.WriteLine($"  {i + 1}: Min: {min}, Max: {max}, Avg: {avg}");
        }
        Console.WriteLine("- Activations:");
        for (int i = 0; i < scalarContext.activations.Length; i++) {
            float min = scalarContext.activations[i].Min();
            float max = scalarContext.activations[i].Max();
            float avg = scalarContext.activations[i].Average();
            Console.WriteLine($"  {i}: Min: {min}, Max: {max}, Avg: {avg}");
        }

        int layers = scalarContext.activations.Length;
        int[] layerSize = new int[layers];
        for (int i = 0; i < layers; i++) {
            layerSize[i] = scalarContext.activations[i].Length;
        }
        
        _simd.ForwardNoise("Scalar",
            CompareLayerArrays(layers, layerSize, scalarContext, simdContext)
        );
        _gpu.ForwardNoise("Scalar",
            CompareLayerArrays(layers, layerSize, scalarContext, gpuContext)
        );
    }
    
    private void CompareBackwardLayers() {
        Console.WriteLine("Comparing layers.");

        BackwardData scalarContext = _scalar.ReadBackwardLayers();
        BackwardData simdContext = _simd.ReadBackwardLayers();
        BackwardData gpuContext = _gpu.ReadBackwardLayers();

        int layers = scalarContext.delta.Length;
        int[] layerSize = new int[layers];
        for (int i = 0; i < layers; i++) {
            layerSize[i] = scalarContext.delta[i].Length;
        }
        
        _simd.BackwardNoise("Scalar",
            CompareLayerArrays(layers, layerSize, scalarContext, simdContext)
        );
        _gpu.BackwardNoise("Scalar",
            CompareLayerArrays(layers, layerSize, scalarContext, gpuContext)
        );
    }

    private static ForwardNoise CompareLayerArrays(int layers, int[] layerSize, ForwardData contextA, ForwardData contextB) {
        ForwardNoise noise = (new float[layers - 1], new float[layers]);
            
        for (int i = 0; i < layers; i++) {
            for (int j = 0; j < layerSize[i]; j++) {
                noise.activations[i] = Math.Abs(contextA.activations[i][j] - contextB.activations[i][j]);
            }
            
            if (i == layers - 1) continue;
            
            for (int j = 0; j < layerSize[i + 1]; j++) {
                noise.values[i] = Math.Abs(contextA.values[i][j] - contextB.values[i][j]);
            }
        }

        return noise;
    }

    private static BackwardNoise CompareLayerArrays(int layers, int[] layerSize, BackwardData contextA, BackwardData contextB) {
        
        // layers = Layers - 1 and layerSize starts from the first index
        
        BackwardNoise noise = (new float[layers], new float[layers], new float[layers]);
            
        for (int i = 0; i < layers; i++) {
            for (int j = 0; j < layerSize[i]; j++) {
                noise.delta[i] = Math.Abs(contextA.delta[i][j] - contextB.delta[i][j]);
            }
            
            for (int j = 0; j < layerSize[i]; j++) {
                noise.deltaWeight[i] = Math.Abs(contextA.deltaWeight[i][j] - contextB.deltaWeight[i][j]);
            }
            
            for (int j = 0; j < layerSize[i]; j++) {
                noise.deltaBias[i] = Math.Abs(contextA.deltaBias[i][j] - contextB.deltaBias[i][j]);
            }
        }

        return noise;
    }
}

internal class TestNetwork {
    private static readonly int[] Layers = [784, 16, 10];
    private const int RandomSeed = 50;
    private const float LearningRate = 0.3f;

    private readonly IBackend _backend;
    private readonly Network _network;
    private readonly VirtualNetwork _virtualNetwork;

    public TestNetwork(IBackend backend) {
        _backend = backend;
        _network = new Network(Layers, backend, new Random(RandomSeed));
        _virtualNetwork = new VirtualNetwork(_network);
    }

    public void TestLoad(DataSample<float> sample) {
        _virtualNetwork.Load(sample);
        int target = (int)_backend.ReadSingle(_virtualNetwork.Target);
        
        Console.WriteLine($"[{_backend.GetType()}] Sample Label: {sample.Label}, Target: {target}");
    }

    public void TestPrediction() {
        int prediction = (int)_backend.ReadSingle(_virtualNetwork.Prediction);
        
        Console.WriteLine($"[{_backend.GetType()}] Prediction: {prediction}");
    }

    public void TestLoss() {
        _virtualNetwork.EvaluateLoss();
        int loss = (int)_backend.ReadSingle(_virtualNetwork.Loss);
        
        Console.WriteLine($"[{_backend.GetType()}] Loss: {loss}");
    }

    public void ForwardNoise(string baseContext, ForwardNoise noise) {
        Console.WriteLine($"[{_backend.GetType()}] Forward Propagation:");
        Console.WriteLine($"- Values:");
        for (int i = 0; i < noise.values.Length; i++) {
            Console.WriteLine($"  {i + 1}: {noise.values[i]}");
        }
        Console.WriteLine($"- Activations:");
        for (int i = 0; i < noise.activations.Length; i++) {
            Console.WriteLine($"  {i}: {noise.activations[i]}");
        }
    }
        
    public void BackwardNoise(string baseContext, BackwardNoise noise) {
        Console.WriteLine($"[{_backend.GetType()}] Backward noise based on {baseContext}:");
        Console.WriteLine($"- Delta:");
        for (int i = 0; i < noise.delta.Length; i++) {
            Console.WriteLine($"  {i + 1}: {noise.delta[i]}");
        }
        Console.WriteLine($"- Delta Weights:");
        for (int i = 0; i < noise.deltaWeight.Length; i++) {
            Console.WriteLine($"  {i + 1}: {noise.deltaWeight[i]}");
        }
        Console.WriteLine($"- Delta Bias:");
        for (int i = 0; i < noise.deltaBias.Length; i++) {
            Console.WriteLine($"  {i + 1}: {noise.deltaBias[i]}");
        }
    }

    public void TestNetworkInitialization() {
        Console.WriteLine($"[{_backend.GetType()}] Network parameters:");

        var networkData = _network.GetData();
        
        Console.WriteLine($"- Weights:");
        for (int i = 0; i < networkData.weights.Length; i++) {
            float min = networkData.weights[i].Min();
            float max = networkData.weights[i].Max();
            float avg = networkData.weights[i].Average();
            Console.WriteLine($"  {i + 1}: Min: {min}, Max: {max}, Avg: {avg}");
        }
        
        Console.WriteLine($"- Bias:");
        for (int i = 0; i < networkData.bias.Length; i++) {
            float min = networkData.bias[i].Min();
            float max = networkData.bias[i].Max();
            float avg = networkData.bias[i].Average();
            Console.WriteLine($"  {i + 1}: Min: {min}, Max: {max}, Avg: {avg}");
        }
    }
    
    public void IterateForward() {
        _virtualNetwork.IterateForward(_network);
    }

    public void IterateBackward() {
        _virtualNetwork.IterateBackward(_network);
    }

    public void IterateAdjust() {
        _virtualNetwork.IterateAdjust(_network, LearningRate);
    }

    public ForwardData ReadForwardLayers() {
        return _virtualNetwork.GetForwardData();
    }

    public BackwardData ReadBackwardLayers() {
        return _virtualNetwork.GetBackwardData();
    }
}