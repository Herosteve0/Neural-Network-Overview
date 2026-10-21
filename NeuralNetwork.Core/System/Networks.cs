using NeuralNetwork.Core.System.Backend;
using NeuralNetwork.Core.System.Data;

namespace NeuralNetwork.Core.System;

using NetworkValues = (float[][] weights, float[][] bias);

/**
 * <param name="Accuracy">
 * The accuracy of the network. In range [0, 1].
 * </param>
 */
public record NetworkInstance(Network Network, float Accuracy) {
    public NetworkInstance Copy() {
        return this with { Network = Network.Copy() };
    }
    
    public static implicit operator Network(NetworkInstance instance) {
        return instance.Network;
    }
    
    public void Dispose() {
        Network.Dispose();
    }
}
    
public class Network : IDisposable{
    private BaseLayer[] Layers { get; }
    public IBackend Backend { get; }
        
    public Network(int[] layers, IBackend backend, Random random) {
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
    public Network(int[] layers, IBackend backend, float[][] weights, float[][] bias) {
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
        
    public NetworkValues GetData() {
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
    public Network Copy() {
        NetworkValues variables = GetData();
        return new Network(LayerLength, Backend, variables.weights, variables.bias);
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
    public readonly IValueFloat Loss;
    
    /**
     * Correct Prediction tracker of the virtual network.
     */
    public readonly IValueInt Correct;
    /**
     * Answer the input should give.
     */
    public readonly IValueInt Target;
    /**
     * Prediction made with given input.
     */
    public readonly IValueInt Prediction;
        
    private readonly ITensor _input;
    private readonly ITensor _expectedOutput;
    private readonly ITensor[] _values;
    private readonly ITensor[] _activations;
        
    private readonly ITensor[] _delta;
    private readonly ITensor[] _weightDelta;
    private readonly ITensor[] _biasDelta;

    private readonly IBackend _backend;
    private readonly int _length;
        
    public VirtualNetwork(Network network) : this(network.LayerLength.ToArray(), network.Backend) { }
    public VirtualNetwork(int[] layers, IBackend backend) {
        _length = layers.Length;
        _backend = backend;

        Correct = _backend.CreateValue(0);
        Loss = _backend.CreateValue(0f);
        Target = _backend.CreateValue(0);
        Prediction = _backend.CreateValue(0);
            
        _values = new ITensor[_length - 1];
        _activations = new ITensor[_length];
        
        _delta = new ITensor[_length - 1];
        _weightDelta = new ITensor[_length - 1];
        _biasDelta = new ITensor[_length - 1];
            
        _input = _backend.Zero(layers[0]);
        _expectedOutput = _backend.Zero(layers[_length - 1]);
            
        for (int i = 0; i < _length; i++) {
            _activations[i] = _backend.Zero(layers[i]);

            if (i == _length - 1) continue;
                
            _values[i] = _backend.Zero(layers[i + 1]);

            _delta[i] = _backend.Zero(layers[i + 1]);
            _weightDelta[i] = _backend.Zero(layers[i + 1], layers[i]);
            _biasDelta[i] = _backend.Zero(layers[i + 1]);
        }
    }
        

    public void Load(DataSample<float> input) {
        _rawInput = input;
        _backend.Set(_input, input.Values);
        _backend.SetSingle(Target, input.Label);

        float[] expected = new float[_length == 0 ? 0 : _expectedOutput.Length];
        expected[input.Label] = 1f;
        _backend.Set(_expectedOutput, expected);
    }

    public DataGuess<float> RawGuess => new DataGuess<float>(
        _rawInput,
        _backend.ReadSingle(Prediction)
    );
    public ITensor Result => _activations[_length - 1];
    public bool IsCorrect => _backend.ReadSingle(Prediction) == _rawInput.Label;

    public (float[][] values, float[][] activations) GetForwardData() {
        float[][] values = new float[_length - 1][];
        for (int i = 0; i < _values.Length; i++) {
            values[i] = _backend.Read(_values[i]);
        }
        float[][] activations = new float[_length][];
        for (int i = 0; i < _activations.Length; i++) {
            activations[i] = _backend.Read(_activations[i]);
        }
        
        return (values, activations);
    }
    public (float[][] delta, float[][] deltaWeight, float[][] deltaBias) GetBackwardData() {
        float[][] delta = new float[_length - 1][];
        for (int i = 0; i < _delta.Length; i++) {
            delta[i] = _backend.Read(_delta[i]);
        }
        float[][] deltaWeight = new float[_length - 1][];
        for (int i = 0; i < _weightDelta.Length; i++) {
            deltaWeight[i] = _backend.Read(_weightDelta[i]);
        }
        float[][] deltaBias = new float[_length - 1][];
        for (int i = 0; i < _biasDelta.Length; i++) {
            deltaBias[i] = _backend.Read(_biasDelta[i]);
        }
        
        return (delta, deltaWeight, deltaBias);
    }

    public void CompleteForwardPass(Network network) {
        IterateForward(network);
        EvaluateAnswer();
        EvaluateLoss();
    }
    
    public void IterateForward(Network network) {
        for (int i = 0; i < network.LayerAmount; i++) {
            Forward(i, network[i]);
        }
    }
    public void IterateBackward(Network network) {
        for (int i = network.LayerAmount - 1; i > 0; i--) {
            Backward(i, network[i]);
        }
    }

    public void IterateAdjust(Network network, float scale) {
        for (int i = 1; i < network.LayerAmount; i++) {
            Adjust(i, network[i], scale);
        }
    }

    public void AdjustCorrect(IValueInt correctOut) {
        _backend.AddSingle(correctOut, Correct);
    }
    
    public void AdjustLoss(IValueFloat lossOut) {
        _backend.AddSingle(lossOut, Loss);
    }

    public void EvaluateAnswer() {
        _backend.CheckAnswer(Prediction, Target, Correct);
    }
    
    public void EvaluateLoss() {
        _backend.Loss(Result, _rawInput.Label, Loss);
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
            
        if (index == _length - 1) {
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
        _backend.Synchronize();
        
        Correct.Dispose();
        Loss.Dispose();
        Prediction.Dispose();
        Target.Dispose();
            
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