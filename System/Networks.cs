using ILGPU.IR.Values;
using Neurtal_Network_Overview.System.Data;

namespace Neurtal_Network_Overview.System;

public record NetworkInstance(NeuralNetwork Network, int Accuracy) {
    public void Dispose() {
        Network.Dispose();
    }
}
    
public class NeuralNetwork : IDisposable{
        
    private BaseLayer[] Layers { get; }
        
    public IBackend Backend { get; }
        
    public NeuralNetwork(int[] layers, IBackend backend, Random random) {
        Backend = backend;
        LayerAmount = layers.Length;
        LayerLength = layers.ToArray();
            
        Layers = new BaseLayer[LayerAmount];
        for (int i = 0; i < LayerAmount; i++) {
            int length = layers[i];
            Layers[i] = i == LayerAmount - 1
                ? new OutputLayer(length, backend, Layers[i - 1], random)
                : i == 0
                    ? new InputLayer(length, backend)
                    : new Layer(length, backend, Layers[i - 1], random);
        }
    }
    public NeuralNetwork(int[] layers, IBackend backend, float[][] weights, float[][] bias) {
        Backend = backend;
        LayerAmount = layers.Length;
        LayerLength = layers.ToArray();
            
        Layers = new BaseLayer[LayerAmount];
        for (int i = 0; i < LayerAmount; i++) {
            int length = layers[i];
            Layers[i] = i == LayerAmount - 1
                ? new OutputLayer(length, backend, Layers[i - 1], weights[i - 1], bias[i - 1])
                : i == 0
                    ? new InputLayer(length, backend)
                    : new Layer(length, backend, Layers[i - 1], weights[i - 1], bias[i - 1]);
        }
    }
        
    public BaseLayer this[int index] => Layers[index];


    public int LayerAmount { get; }
    public int[] LayerLength { get; }

    public DataGuess<float> QuickCalculate(DataSample<float> input) {
        VirtualNetwork network = new VirtualNetwork(this);
        Calculate(input, network);
        DataGuess<float> result = network.RawGuess;
        network.Dispose();
        return result;
    }
    /**
     * Fills the given network with the correct values, guess index and loss, appropriate to the input.
     * <returns>
     * Returns the output tensor
     * </returns>
     */
    public ITensor Calculate(DataSample<float> input, VirtualNetwork network) {
        network.Load(input);
        network.CompleteForwardPass(this);
        return network.Result;
    }

    public void UpdateTranspose() {
        for (int i = 1; i < LayerAmount; i++) {
            ((Layer)this[i]).UpdateTranspose();
        }
    }
        
    public (float[][] weights, float[][] bias) GetData() {
        float[][] weights = new float[LayerAmount - 1][];
        float[][] bias = new float[LayerAmount - 1][];
            
        for (int i = 1; i < LayerAmount; i++) {
            weights[i - 1] = new float[LayerLength[i] * LayerLength[i - 1]];
            Backend.Read(((Layer)this[i]).Weights).CopyTo(weights[i - 1]);
                
            bias[i - 1] = new float[LayerLength[i]];
            Backend.Read(((Layer)this[i]).Bias).CopyTo(bias[i - 1]);
        }
            
        return (weights, bias);
    }
    public NeuralNetwork Copy() {
        var variables = GetData();
        return new NeuralNetwork(LayerLength, Backend, variables.weights, variables.bias);
    }

    public void Dispose() {
        Backend.Synchronize();
        foreach (BaseLayer layer in Layers) {
            layer.Dispose();
        }
    }
}

public class VirtualNetwork : IDisposable {
    private DataSample<float> _rawInput;
        
    /**
     * Loss of the given evaluation.
     */
    public readonly ISingleValue Loss;
    /**
     * Answer the input should give.
     */
    public readonly ISingleValue Target;
    /**
     * Prediction made with given input.
     */
    public readonly ISingleValue Prediction;
        
    private readonly ITensor _input;
    private readonly ITensor _expectedOutput;
    private readonly ITensor[] _values;
    private readonly ITensor[] _activations;
        
    private readonly ITensor[] _delta;
    private readonly ITensor[] _weightDelta;
    private readonly ITensor[] _biasDelta;

    public readonly IBackend Backend;
    public readonly int Length;
        
    public VirtualNetwork(NeuralNetwork network) : this(network.LayerLength.ToArray(), network.Backend) { }
    public VirtualNetwork(int[] layers, IBackend backend) {
        Length = layers.Length;
        Backend = backend;

        Loss = Backend.CreateValue(0);
        Target = Backend.CreateValue(0);
        Prediction = Backend.CreateValue(0);
            
        _values = new ITensor[Length - 1];
        _activations = new ITensor[Length];
            
        _delta = new ITensor[Length - 1];
        _weightDelta = new ITensor[Length - 1];
        _biasDelta = new ITensor[Length - 1];
            
        _input = Backend.Zero(layers[0]);
        _expectedOutput = Backend.Zero(layers[Length - 1]);
            
        for (int i = 0; i < Length; i++) {
            _activations[i] = Backend.Zero(layers[i]);

            if (i == Length - 1) continue;
                
            _values[i] = Backend.Zero(layers[i + 1]);

            _delta[i] = Backend.Zero(layers[i + 1]);
            _weightDelta[i] = Backend.Zero(layers[i + 1], layers[i]);
            _biasDelta[i] = Backend.Zero(layers[i + 1]);
        }
    }
        

    public void Load(DataSample<float> input) {
        _rawInput = input;
        Backend.Set(_input, input.Values);
        Backend.SetSingle(Target, input.Label);

        float[] expected = new float[Length == 0 ? 0 : _expectedOutput.Length];
        expected[input.Label] = 1f;
        Backend.Set(_expectedOutput, expected);
    }

    public DataGuess<float> RawGuess => new DataGuess<float>(
        _rawInput,
        (int)Backend.ReadSingle(Prediction)
    );
    public ITensor Result => _activations[Length - 1];
    public bool IsCorrect => (int)Backend.ReadSingle(Target) == (int)Backend.ReadSingle(Prediction);

    public (float[][] values, float[][] activations) GetForwardData() {
        float[][] values = new float[Length - 1][];
        for (int i = 0; i < _values.Length; i++) {
            values[i] = Backend.Read(_values[i]);
        }
        float[][] activations = new float[Length][];
        for (int i = 0; i < _activations.Length; i++) {
            activations[i] = Backend.Read(_activations[i]);
        }
        
        return (values, activations);
    }
    public (float[][] delta, float[][] deltaWeight, float[][] deltaBias) GetBackwardData() {
        float[][] delta = new float[Length - 1][];
        for (int i = 0; i < _delta.Length; i++) {
            delta[i] = Backend.Read(_delta[i]);
        }
        float[][] deltaWeight = new float[Length - 1][];
        for (int i = 0; i < _weightDelta.Length; i++) {
            deltaWeight[i] = Backend.Read(_weightDelta[i]);
        }
        float[][] deltaBias = new float[Length - 1][];
        for (int i = 0; i < _biasDelta.Length; i++) {
            deltaBias[i] = Backend.Read(_biasDelta[i]);
        }
        
        return (delta, deltaWeight, deltaBias);
    }

    public void CompleteForwardPass(NeuralNetwork network) {
        IterateForward(network);
        EvaluateLoss();
    }
    
    public void IterateForward(NeuralNetwork network) {
        for (int i = 0; i < network.LayerAmount; i++) {
            Forward(i, network[i]);
        }
    }
    public void IterateBackward(NeuralNetwork network) {
        for (int i = network.LayerAmount - 1; i > 0; i--) {
            Backward(i, network[i]);
        }
    }

    public void IterateAdjust(NeuralNetwork network, float scale, ISingleValue loss) {
        if (loss != null) Backend.AddSingle(loss, Loss);
        for (int i = 1; i < network.LayerAmount; i++) {
            Adjust(i, network[i], scale);
        }
    }

    public void EvaluateLoss() {
        Backend.Loss(Result, _rawInput.Label, Loss);
    }

    public void Forward(int index, BaseLayer layer) {
        // Layer 0 has no parameters, so index is shifted by -1
        int l = index - 1;

        if (index == 0) {
            layer.Forward(_input, null, _activations[index], Prediction);
        } else {
            layer.Forward(_activations[index - 1], _values[l], _activations[index], Prediction);
        }
    }

    public void Backward(int index, BaseLayer layer) {
        // Layer 0 has no parameters, so index is shifted by -1
        int l = index - 1;
            
        if (index == Length - 1) {
            layer.Backward(_activations[index], _expectedOutput, _delta[l]);
        } else {
            layer.Backward(_values[l], _delta[l + 1] , _delta[l]);
        }
            
        layer.BackPropagation(_delta[l], _activations[index - 1], _weightDelta[l], _biasDelta[l]);
    }

    public void Adjust(int index, BaseLayer layer, float scale) {
        // Layer 0 has no parameters, so index is shifted by -1
        int l = index - 1;
            
        layer.Adjust(_weightDelta[l], _biasDelta[l], scale);
    }

    public void Dispose() {
        Backend.Synchronize();
        
        Loss.Dispose();
        Prediction.Dispose();
            
        _input.Dispose();
        _expectedOutput.Dispose();
        foreach (ITensor tensor in _values) {
            tensor.Dispose();
        }
        foreach (ITensor tensor in _activations) {
            tensor.Dispose();
        }
        foreach (ITensor tensor in _delta) {
            tensor.Dispose();
        }
        foreach (ITensor tensor in _weightDelta) {
            tensor.Dispose();
        }
        foreach (ITensor tensor in _biasDelta) {
            tensor.Dispose();
        }
    }
}