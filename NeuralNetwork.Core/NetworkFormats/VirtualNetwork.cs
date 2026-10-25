using NeuralNetwork.Core.Backend;
using NeuralNetwork.Core.Data;
using NeuralNetwork.Core.Data.SingleValues;
using NeuralNetwork.Core.Data.Tensors;
using NeuralNetwork.Core.NetworkFormats.Layers;

namespace NeuralNetwork.Core.NetworkFormats;

public class VirtualNetwork : IDisposable {
    /**
     * Loss of the given evaluation.
     */
    private readonly ISingleValue<float> _loss;
    /**
     * Total Loss of the given buffer.
     */
    private readonly ISingleValue<float> _bufferLoss;

    /**
     * Correct Prediction tracker of the virtual network.
     */
    private readonly ISingleValue<int> _correct;
    
    /**
     * Answer the input should give.
     */
    private readonly ISingleValue<int>[] _targets;
    /**
     * Prediction made with given input.
     */
    private readonly ISingleValue<int>[] _predictions;
        
    private DataBatch<float> _inputBuffer;
    private readonly ITensor<float>[] _input;
    private readonly ITensor<float>[] _expectedOutput;
    
    private readonly ITensor<float>[] _values;
    private readonly ITensor<float>[] _activations;
        
    private readonly ITensor<float>[] _delta;
    private readonly ITensor<float>[] _weightDelta;
    private readonly ITensor<float>[] _biasDelta;

    private readonly int _bufferSize;
    private int _bufferIndex;
    
    private readonly IBackend _backend;
    private readonly int _length;

    public VirtualNetwork(Network network, int bufferSize = 1) : this(network.LayerLength.ToArray(), network.Backend, bufferSize) { }
    public VirtualNetwork(int[] layers, IBackend backend, int bufferSize) {
        _bufferSize = bufferSize;
        _length = layers.Length;
        _backend = backend;

        _loss = _backend.CreateValue(0f);
        _bufferLoss = _backend.CreateValue(0f);
        _correct = _backend.CreateValue(0);
        
        _targets = new ISingleValue<int>[bufferSize];
        _predictions = new ISingleValue<int>[bufferSize];
        _input = new ITensor<float>[bufferSize];
        _expectedOutput = new ITensor<float>[bufferSize];

        for (int i = 0; i < bufferSize; i++) {
            _targets[i] = _backend.CreateValue(0);
            _predictions[i] = _backend.CreateValue(0);
            
            _input[i] = _backend.CreateTensor(0f, layers[0]);
            _expectedOutput[i] = _backend.CreateTensor(0f, layers[_length - 1]);
        }
        
        _values = new ITensor<float>[_length - 1];
        _activations = new ITensor<float>[_length];

        _delta = new ITensor<float>[_length - 1];
        _weightDelta = new ITensor<float>[_length - 1];
        _biasDelta = new ITensor<float>[_length - 1];
        
        for (int i = 0; i < _length; i++) {
            _activations[i] = _backend.CreateTensor(0f, layers[i]);

            if (i == _length - 1) continue;
                
            _values[i] = _backend.CreateTensor(0f, layers[i + 1]);

            _delta[i] = _backend.CreateTensor(0f, layers[i + 1]);
            _weightDelta[i] = _backend.CreateTensor(0f, layers[i + 1], layers[i]);
            _biasDelta[i] = _backend.CreateTensor(0f, layers[i + 1]);
        }
    }


    public void LoadBuffer(DataBatch<float> input) {
        if (input.Size > _bufferSize)
            throw new ArgumentException($"Tried to cache more samples than expected. {input.Size} instead of {_bufferSize}.", nameof(input));
        
        _bufferIndex = 0;
        _inputBuffer = input;
        
        _backend.SetSingle(_bufferLoss, 0f);
        
        for (int i = 0; i < _inputBuffer.Size; i++) {
            _backend.Set(_input[i], input[i].Values);
            _backend.SetSingle(_targets[i], input[i].Label);
            
            float[] expected = new float[_length == 0 ? 0 : _expectedOutput[i].Length];
            expected[input[i].Label] = 1f;
            _backend.Set(_expectedOutput[i], expected);
        }
    }

    public DataGuess<float> RawGuess(int index) {
        return new DataGuess<float>(
            _inputBuffer[index], 
            _backend.ReadSingle(_predictions[index]
            )
        );
    }

    public bool[] IsCorrect =>
        Enumerable.Range(0, _inputBuffer.Size).Select(i => 
            _backend.ReadSingle(_predictions[i]) == _inputBuffer[i].Label
        ).ToArray();
    
    private ITensor<float> Result => _activations[_length - 1];

    private ISingleValue<int> CurrentPrediction => _predictions[_bufferIndex];
    private ISingleValue<int> CurrentTarget => _targets[_bufferIndex];
    
    private ITensor<float> CurrentInput => _input[_bufferIndex];
    private ITensor<float> CurrentExpectedOutput => _expectedOutput[_bufferIndex];
    
    
    /**
     * Processes all the samples currently in the buffer.
     */
    public void ProcessFullBuffer(Action process) {
        for (; _bufferIndex < _inputBuffer.Size; _bufferIndex++) {
            process();
        }
    }
    
    public void CompleteForwardPass(Network network) {
        IterateForward(network);
        EvaluateAnswer();
        EvaluateLoss();
    }

    private void IterateForward(Network network) {
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

    public void AdjustCorrect(ISingleValue<int> correctOut) {
        _backend.AddSingle(correctOut, _correct);
    }

    public void AdjustLoss(ISingleValue<float> lossOut) {
        _backend.AddSingle(lossOut, _bufferLoss);
    }

    private void EvaluateAnswer() {
        _backend.CheckAnswer(CurrentPrediction, CurrentTarget, _correct);
    }

    private void EvaluateLoss() {
        _backend.Loss(Result, CurrentTarget, _loss);
        _backend.AddSingle(_bufferLoss, _loss);
    }

    private void Forward(int index, BaseLayer layer) {
        // Layer 0 has no parameters, so index is shifted by -1
        int l = index - 1;

        if (index == 0) {
            layer.Forward(CurrentInput, null, _activations[index], CurrentPrediction);
        } else {
            layer.Forward(_activations[index - 1], _values[l], _activations[index], CurrentPrediction);
        }
    }

    private void Backward(int index, BaseLayer layer) {
        // Layer 0 has no parameters, so index is shifted by -1
        int l = index - 1;
        
        if (index == _length - 1) {
            layer.Backward(_activations[index], CurrentExpectedOutput, _delta[l]);
        } else {
            layer.Backward(_values[l], _delta[l + 1] , _delta[l]);
        }
            
        layer.BackPropagation(_delta[l], _activations[index - 1], _weightDelta[l], _biasDelta[l]);
    }

    private void Adjust(int index, BaseLayer layer, float scale) {
        // Layer 0 has no parameters, so index is shifted by -1
        int l = index - 1;
            
        layer.Adjust(_weightDelta[l], _biasDelta[l], scale);
    }

    public void Dispose() {
        _backend.Synchronize();
        
        _correct.Dispose();
        _loss.Dispose();
        
        foreach (ISingleValue<int> value in _predictions) {
            value.Dispose();
        }
        foreach (ISingleValue<int> value in _targets) {
            value.Dispose();
        }

        foreach (ITensor<float> tensor in _input) {
            tensor.Dispose();
        }
        foreach (ITensor<float> tensor in _expectedOutput) {
            tensor.Dispose();
        }
        
        foreach (ITensor<float> tensor in _values) {
            tensor.Dispose();
        }
        foreach (ITensor<float> tensor in _activations) {
            tensor.Dispose();
        }
        foreach (ITensor<float> tensor in _delta) {
            tensor.Dispose();
        }
        foreach (ITensor<float> tensor in _weightDelta) {
            tensor.Dispose();
        }
        foreach (ITensor<float> tensor in _biasDelta) {
            tensor.Dispose();
        }
    }
}