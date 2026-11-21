using NeuralNetwork.Core.Backend;
using NeuralNetwork.Core.Data.SingleValues;
using NeuralNetwork.Core.Data.Tensors;
using NeuralNetwork.Core.NetworkFormats.Layers.State;

namespace NeuralNetwork.Core.NetworkFormats.Layers;

public interface ILayer {
    /**
     * Creates a pass state used for instancing the network.
     */
    ILayerState CreatePassState();
    
    /**
     * This layer prepares the input for the next layer's inputs. If it's the last layer, makes a prediction instead.
     */
    void Forward(ITensor<float> input, ILayerState passState, ISingleValue<int> predictionOut, PassMode mode);
    /**
     * This layer prepares the previous layer's delta using its own. The Loss Function should be used for the last layer.
     */
    void Delta(ILayerStateDelta backwardState, ITensor<float> deltaOut);
}

public interface ITrainableLayer {
    /**
     * This layer applies it's activation function derivative to its delta.
     */
    void ActivationDelta(ILayerStateDelta passState);
    void BackPropagation(ITensor<float> activations, ILayerStateDelta passState);
    void Adjust(ILayerState passState, float scale);
    
    void Mutate(Random random, float mutationStrength);
}

public abstract class BaseLayer(int size) : IDisposable {
    public int NeuronCount { get; protected set; } = size;
    protected IBackend Backend { get; private set; }
    private bool _initialized;
    protected bool Disposed;

    public void Initialize(IBackend backend) {
        ArgumentNullException.ThrowIfNull(backend);
        if (_initialized)
            throw new InvalidOperationException("A layer can only be initialized once.");
        
        Backend = backend;
        _initialized = true;
    }
    
    public virtual void Initialize(int inputSize, IBackend backend, Random random) {
        Initialize(backend);
    }

    public virtual void InitializeFromData(int inputSize, IBackend backend, float[][] data) {
        Initialize(backend);
    }
    
    protected static TPass TranslateState<TPass>(ILayerState passState) where TPass : ILayerState {
        if (passState is not TPass state)
            throw new InvalidDataException(
                $"The pass state must be {typeof(TPass).Name}, got {passState.GetType().Name}.");
        return state;
    }
    protected static TPass TranslateState<TPass>(ILayerStateDelta passState) where TPass : ILayerState {
        if (passState is not TPass state)
            throw new InvalidDataException(
                $"The pass state must be {typeof(TPass).Name}, got {passState.GetType().Name}.");
        return state;
    }

    /**
     * Returns an initialized copy of the layer.
     */
    public abstract BaseLayer Copy();
    
    /**
     * Saves the layer inside a given writer.
     */
    public abstract void Save(BinaryWriter writer);
    
    /**
     * Loads a layer from a reader, initialized with a given backend.
     */
    public static BaseLayer Load(BinaryReader reader, IBackend backend) {
        string type = reader.ReadString();

        return type switch {
            "input" => InputLayer.Load(reader, backend),
            "dropout" => DropoutLayer.Load(reader, backend),
            "fully_connected" => FullyConnectedLayer.Load(reader, backend),
            _ => throw new InvalidDataException($"Unknown layer type '{type}' in network data.")
        };
    }
    
    public abstract void Dispose();
    
    protected void ThrowIfDisposed() {
        ObjectDisposedException.ThrowIf(Disposed, this);
    }
}