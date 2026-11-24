using NeuralNetwork.Core.Backend;
using NeuralNetwork.Core.Data;
using NeuralNetwork.Core.Data.SingleValues;
using NeuralNetwork.Core.Data.Tensors;
using NeuralNetwork.Core.NetworkFormats.Networks;

namespace NeuralNetwork.Core.NetworkFormats.ExecutableNetworks;

public abstract class VirtualNetwork : IDisposable {
    protected DataBatch<float> InputBuffer;
    private readonly ITensor<float>[] _input;
    private readonly ISingleValue<int>[] _targets;
    
    protected readonly int BufferSize;
    protected int BufferIndex;
    
    protected readonly IBackend Backend;
    protected readonly int Length;
    private bool _disposed;

    protected VirtualNetwork(Network network, int bufferSize = 1) {
        ArgumentNullException.ThrowIfNull(network);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(bufferSize);
        
        BufferSize = bufferSize;
        Length = network.LayerAmount;
        Backend = network.Backend;
        
        _input = new ITensor<float>[bufferSize];
        _targets = new ISingleValue<int>[bufferSize];

        for (int i = 0; i < bufferSize; i++) {
            _input[i] = Backend.CreateTensor(0f, network.LayerLength[0]);
            _targets[i] = Backend.CreateValue(0);
        }
    }


    public void LoadBuffer(DataBatch<float> input) {
        ArgumentNullException.ThrowIfNull(input);
        ObjectDisposedException.ThrowIf(_disposed, this);
        
        if (input.Size == 0)
            throw new ArgumentException("Cannot load an empty buffer.", nameof(input));
        if (input.Size > BufferSize)
            throw new ArgumentException($"Tried to cache more samples than expected. {input.Size} instead of {BufferSize}.", nameof(input));
        
        BufferIndex = 0;
        InputBuffer = input;
        
        FillBuffer(input);
    }

    protected virtual void FillBuffer(DataBatch<float> input) {
        for (int i = 0; i < InputBuffer.Size; i++) {
            DataSample<float> sample = input[i];
            if (sample.Values == null || sample.Values.Length != _input[i].Length)
                throw new ArgumentException($"Sample {i} must contain exactly {_input[i].Length} input values.", nameof(input));
            
            Backend.Set(_input[i], sample.Values);
            Backend.Set(_targets[i], sample.Label);
        }
    }

    protected ITensor<float> CurrentInput => _input[BufferIndex];
    protected ISingleValue<int> CurrentTarget => _targets[BufferIndex];

    /**
     * Processes all the samples currently in the buffer.
     */
    public void ProcessFullBuffer(Action process) {
        for (; BufferIndex < InputBuffer.Size; BufferIndex++) {
            process();
        }
    }

    public abstract void CompleteForwardPass(Network network);

	public void Dispose() {
		Dispose(true);
		GC.SuppressFinalize(this);
	}
	
	protected virtual void Dispose(bool disposing) {
		if (_disposed) return;
        
        Backend.Synchronize();

        foreach (ITensor<float> tensor in _input) {
            tensor.Dispose();
        }
        foreach (ISingleValue<int> value in _targets) {
            value.Dispose();
        }
        
		_disposed = true;
	}
}