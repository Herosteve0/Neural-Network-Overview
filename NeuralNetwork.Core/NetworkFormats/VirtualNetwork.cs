using NeuralNetwork.Core.Backend;
using NeuralNetwork.Core.Data;
using NeuralNetwork.Core.Data.SingleValues;
using NeuralNetwork.Core.Data.Tensors;
using NeuralNetwork.Core.NetworkFormats.Layers;
using NeuralNetwork.Core.NetworkFormats.Layers.State;

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

    private readonly ILayerState[] _states;
    
    private readonly int _bufferSize;
    private int _bufferIndex;
    
    private readonly IBackend _backend;
    private readonly int _length;
    private bool _disposed;

    public VirtualNetwork(Network network, int bufferSize = 1) {
        ArgumentNullException.ThrowIfNull(network);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(bufferSize);

        _bufferSize = bufferSize;
        _length = network.LayerAmount;
        _backend = network.Backend;
        _lossFunction = network.LossFunction;

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
            
            _input[i] = _backend.CreateTensor(0f, network.LayerLength[0]);
            _expectedOutput[i] = _backend.CreateTensor(0f, network.LayerLength[_length - 1]);
        }

        _states = new ILayerState[_length];
        for (int i = 0; i < _length; i++) {
            _states[i] = network[i].CreatePassState();
        }
    }


    public void LoadBuffer(DataBatch<float> input) {
        ArgumentNullException.ThrowIfNull(input);
        ObjectDisposedException.ThrowIf(_disposed, this);
        
        if (input.Size == 0)
            throw new ArgumentException("Cannot load an empty buffer.", nameof(input));
        if (input.Size > _bufferSize)
            throw new ArgumentException($"Tried to cache more samples than expected. {input.Size} instead of {_bufferSize}.", nameof(input));
        
        _bufferIndex = 0;
        _inputBuffer = input;
        
        _backend.SetSingle(_bufferLoss, 0f);
        
        for (int i = 0; i < _inputBuffer.Size; i++) {
            DataSample<float> sample = input[i];
            if (sample.Values == null || sample.Values.Length != _input[i].Length)
                throw new ArgumentException($"Sample {i} must contain exactly {_input[i].Length} input values.", nameof(input));
            if ((uint)sample.Label >= (uint)_expectedOutput[i].Length)
                throw new ArgumentOutOfRangeException(nameof(input), $"Sample {i} has label {sample.Label}, but the output layer has {_expectedOutput[i].Length} neurons.");
            
            _backend.Set(_input[i], sample.Values);
            _backend.SetSingle(_targets[i], sample.Label);
            _backend.Set(_expectedOutput[i], sample.LabelArray(_expectedOutput[i].Length, 1f));
        }
    }

    public DataGuess<float> RawGuess(int index) {
        return new DataGuess<float>(
            _inputBuffer[index], 
            _backend.ReadSingle(_predictions[index])
        );
    }

    public bool[] IsCorrect =>
        Enumerable.Range(0, _inputBuffer.Size).Select(i => 
            _backend.ReadSingle(_predictions[i]) == _inputBuffer[i].Label
        ).ToArray();
    
    private ITensor<float> Result => _states[_length - 1].Output;

    private ISingleValue<int> CurrentPrediction => _predictions[_bufferIndex];
    private ISingleValue<int> CurrentTarget => _targets[_bufferIndex];
    
    private ITensor<float> CurrentInput => _input[_bufferIndex];
    private ITensor<float> CurrentExpectedOutput => _expectedOutput[_bufferIndex];

    private readonly ILoss _lossFunction;
    
    /**
     * Processes all the samples currently in the buffer.
     */
    public void ProcessFullBuffer(Action process) {
        for (; _bufferIndex < _inputBuffer.Size; _bufferIndex++) {
            process();
        }
    }
    
    public void CompleteForwardPass(Network network, PassMode mode) {
        IterateForward(network, mode);
        EvaluateAnswer();
        EvaluateLoss();
    }

    public void IterateForward(Network network, PassMode mode) {
        for (int i = 0; i < network.LayerAmount; i++) {
            network[i].Forward(
                i == 0
                    ? CurrentInput
                    : _states[i - 1].Output,
                _states[i],
                i == network.LayerAmount - 1
                    ? CurrentPrediction
                    : null,
                mode
            );
        }
    }

    public void IterateBackward(Network network) {
        for (int i = network.LayerAmount - 1; i > 0; i--) {
            if (_states[i] is not ILayerStateDelta backwardState)
                throw new InvalidDataException($"Layer Forward State {_states[i].GetType().Name} does not implement ILayerBackwardState.");
        
            if (i == _length - 1) {
                _lossFunction.Backward(_states[i].Output, CurrentExpectedOutput, backwardState.Delta);
            } else {
                ILayerStateDelta nextState = (ILayerStateDelta)_states[i + 1];
                network[i + 1].Delta(nextState, backwardState.Delta);
            }

            if (network[i] is not ITrainableLayer layer) continue;

            if (i != _length - 1) {
                layer.ActivationDelta(backwardState);
            }
            
            layer.BackPropagation(_states[i - 1].Output, backwardState);
        }
    }

    public void IterateAdjust(Network network, float scale) {
        for (int i = 1; i < network.LayerAmount; i++) {
            if (network[i] is not ITrainableLayer layer) continue;
            layer.Adjust(_states[i], scale);
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
        _lossFunction.Calculate(Result, CurrentTarget, _loss);
        _backend.AddSingle(_bufferLoss, _loss);
    }

    public void Dispose() {
        if (_disposed) return;
        _disposed = true;
        GC.SuppressFinalize(this);
        
        _backend.Synchronize();
        
        _correct.Dispose();
        _loss.Dispose();
        _bufferLoss.Dispose();
        
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

        foreach (ILayerState state in _states) {
            state.Dispose();
        }
    }
}