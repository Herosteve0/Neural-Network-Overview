using NeuralNetwork.Core.Backend;
using NeuralNetwork.Core.Data;
using NeuralNetwork.Core.NetworkFormats.Layers;
using NeuralNetwork.Core.NetworkFormats.Layers.State;

namespace NeuralNetwork.Core.NetworkFormats;

/**
 * <param name="Accuracy">
 * The accuracy of the network. In range [0, 1].
 * </param>
 */
public record NetworkInstance(Network Network, float Accuracy) : IDisposable {
    public NetworkInstance Copy() {
        return this with { Network = Network.Copy() };
    }
    
    public static implicit operator Network(NetworkInstance instance) {
        return instance.Network;
    }
    
    public void Dispose() {
        Network.Dispose();
        GC.SuppressFinalize(this);
    }
}
    
public class Network : IDisposable {
    public IBackend Backend { get; }
    public ILoss LossFunction { get; }
    private bool _disposed;
    
    private readonly BaseLayer[] _layers;
    public IReadOnlyList<BaseLayer> Layers { get; }
    public int LayerAmount { get; }
    public IReadOnlyList<int> LayerLength { get; }

    public Network(IBackend backend, ILoss lossFunction, Random random, params BaseLayer[] layers) 
        : this(backend, lossFunction, layers, random, true) { }

    private Network(IBackend backend, ILoss lossFunction, BaseLayer[] layers, Random random, bool initializeLayers) {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(lossFunction);
        ArgumentNullException.ThrowIfNull(layers);
        ValidateLayerDefinitions(layers, initializeLayers);
        
        if (initializeLayers && random is null)
            throw new ArgumentNullException(nameof(random));
        
        Backend = backend;
        LossFunction = lossFunction;
        _layers = (BaseLayer[])layers.Clone();
        Layers = Array.AsReadOnly(_layers);
        LayerAmount = _layers.Length;

        int[] layerLengths = new int[LayerAmount];
        LayerLength = Array.AsReadOnly(layerLengths);

        if (initializeLayers) {
            int initializedCount = 0;
            try {
                _layers[0].Initialize(backend);
                initializedCount = 1;
                layerLengths[0] = _layers[0].NeuronCount;

                for (int i = 1; i < LayerAmount; i++) {
                    BaseLayer layer = _layers[i];
                    if (layer is not DropoutLayer && layer.NeuronCount <= 0)
                        throw new ArgumentException($"Layer {i} must have a positive neuron count.", nameof(layers));

                    initializedCount = i + 1;
                    layer.Initialize(layerLengths[i - 1], backend, random);
                    if (layer.NeuronCount <= 0)
                        throw new ArgumentException($"Layer {i} resolved to an invalid neuron count.", nameof(layers));

                    layer.ValidateLayer(i, layerLengths[i - 1]);
                    layerLengths[i] = layer.NeuronCount;
                }
                
                LossFunction.Initialize(backend);
            } catch {
                for (int i = 0; i < initializedCount; i++) {
                    try {
                        _layers[i].Dispose();
                    } catch { /* Ignored */ }
                }
                throw;
            }
        } else {
            for (int i = 0; i < LayerAmount; i++) {
                BaseLayer layer = layers[i];
                if (layer.NeuronCount <= 0)
                    throw new InvalidDataException($"Serialized layer {i} has an invalid neuron count.");
                
                layerLengths[i] = layer.NeuronCount;
                if (i > 0) layer.ValidateLayer(i, layerLengths[i - 1]);
            }
            
            LossFunction.Initialize(backend);
        }
    }

    internal static Network FromLoadedLayers(IBackend backend, ILoss lossFunction, BaseLayer[] layers) {
        return new Network(backend, lossFunction, layers, null, false);
    }

    private static void ValidateLayerDefinitions(BaseLayer[] layers, bool loading) {
        // Layer Count validation
        if (layers.Length < 2)
            throw new ArgumentException("A network must contain at least 2 layers.", nameof(layers));
        // Layer null validation
        if (layers.Any(layer => layer is null))
            throw new ArgumentException("Layers cannot be null.", nameof(layers));
        
        // Input layer checks
        if (layers[0] is not InputLayer)
            throw new ArgumentException("The first layer of a network must be an InputLayer.", nameof(layers));
        if (layers.Skip(1).Any(layer => layer is InputLayer))
            throw new ArgumentException("An input layer must only be at index 0 of a network.", nameof(layers));

        // Ruleset check (or interface)
        if (layers.Any(layer => layer is not ILayer))
            throw new ArgumentException("Every layer must implement from ILayer.", nameof(layers));

        // Instance check
        if (layers.Distinct(ReferenceEqualityComparer.Instance).Count() != layers.Length)
            throw new ArgumentException("A layer instance cannot appear more than once inside the same network.",
                nameof(layers));

        // Input size check
        if (layers[0].NeuronCount <= 0) {
            throw loading
                ? new InvalidDataException("Serialized input layer has an invalid neuron count.")
                : new ArgumentException("Input layer size must be greater than zero.", nameof(layers));
        }
    }

    public ILayer this[int index] => _layers[index] as ILayer
        ?? throw new InvalidOperationException($"Layer {index} does not implement ILayer.");
    
    public DataGuess<float> QuickCalculate(DataSample<float> input) {
        ThrowIfDisposed();
        
        using VirtualNetwork virtualNetwork = new VirtualNetwork(this);
        virtualNetwork.LoadBuffer(new DataBatch<float>(input));
        virtualNetwork.CompleteForwardPass(this, PassMode.Inference);
        return virtualNetwork.RawGuess(0);
    }

    public void Save(BinaryWriter writer) {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(writer);
        
        Backend.Synchronize();
        writer.Write(LayerAmount);
        foreach (BaseLayer layer in _layers) {
            layer.Save(writer);
        }
    }

    public static Network Load(BinaryReader reader, IBackend backend, ILoss lossFunction) {
        ArgumentNullException.ThrowIfNull(reader);
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(lossFunction);

        int layerAmount = reader.ReadInt32();
        if (layerAmount < 2)
            throw new InvalidDataException($"Invalid layer count: {layerAmount}.");
        
        BaseLayer[] layers = new BaseLayer[layerAmount];
        int loaded = 0;

        try {
            for (; loaded < layers.Length; loaded++) {
                layers[loaded] = BaseLayer.Load(reader, backend);
            }

            return FromLoadedLayers(backend, lossFunction, layers);
        } catch {
            for (int i = 0; i < loaded; i++) {
                try {
                    layers[i].Dispose();
                } catch { /* ignored */ }
            }
            throw;
        }
    }
    
    /**
     * Copies this network into a new, separate instance with independent layers.
     */
    public Network Copy() {
        ThrowIfDisposed();
        
        return new Network(
            Backend,
            LossFunction,
            _layers.Select(layer => layer.Copy()).ToArray(),
            null,
            false
        );
    }
    
    /**
     * Mutates this network instance.
     */
    public void Mutate(Random random, float mutationStrength) {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(random);
        if (!float.IsFinite(mutationStrength) || mutationStrength < 0) {
            throw new ArgumentOutOfRangeException(nameof(mutationStrength));
        }
        if (mutationStrength == 0f) return;
        
        foreach (BaseLayer layer in _layers) {
            if (layer is not ITrainableLayer trainableLayer) continue;
            trainableLayer.Mutate(random, mutationStrength);
        }
    }

    private void ThrowIfDisposed() {
        ObjectDisposedException.ThrowIf(_disposed, this);
    }
    
    public void Dispose() {
        if (_disposed) return;
        
        Backend.Synchronize();
        foreach (BaseLayer layer in _layers) {
            layer.Dispose();
        }

        _disposed = true;
        GC.SuppressFinalize(this);
    }
}