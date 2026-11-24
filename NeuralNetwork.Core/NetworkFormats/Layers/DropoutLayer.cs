using NeuralNetwork.Core.Backend;
using NeuralNetwork.Core.Data.SingleValues;
using NeuralNetwork.Core.Data.Tensors;
using NeuralNetwork.Core.NetworkFormats.Layers.State;

namespace NeuralNetwork.Core.NetworkFormats.Layers;

public class DropoutLayer(float dropoutChance, int seed) : BaseLayer(0), ILayer {
    private readonly float _dropoutChance = ValidateChance(dropoutChance);
    private readonly Random _seedRandom = new Random(seed);

    private IElementwiseBackend ElementwiseBackend => Backend as IElementwiseBackend
                                                      ?? throw new NotSupportedException("The selected backend does not implement parameter adjustments.");
    private IRandomBackend RandomBackend => Backend as IRandomBackend
                                                      ?? throw new NotSupportedException("The selected backend does not implement parameter adjustments.");
    
    private static float ValidateChance(float chance) {
        if (!float.IsFinite(chance) || chance is < 0f or >= 1f)
            throw new ArgumentOutOfRangeException(nameof(chance), chance, $"Dropout must be in range [0, 1).");
        return chance;
    }

    public override void Initialize(int inputSize, IBackend backend, Random random) {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(inputSize);
        Initialize(backend);
        
        NeuronCount = inputSize;
    }

    public ILayerForwardState CreateForwardState() {
        return new DropoutForwardState(Backend, NeuronCount, _seedRandom.Next());
    }

    public ILayerBackwardState CreateBackwardState() {
        return new DropoutBackwardState(Backend, NeuronCount);
    }

    public void Forward(
        ITensor<float> input, ILayerForwardState forwardState, ISingleValue<int> predictionOut, PassMode mode
    ) {
        ThrowIfDisposed();
        DropoutForwardState forward = TranslateState<DropoutForwardState>(forwardState);

        if (mode == PassMode.Inference) {
            Backend.Copy(input, forward.Output);
            return;
        }
        
        RandomBackend.GenerateDropoutMask(forward.Random, _dropoutChance, forward.Mask);
        ElementwiseBackend.Dropout(input, forward.Mask, forward.Output);
    }

    public void Delta(ILayerForwardState forwardState, ILayerBackwardState backwardState, ITensor<float> deltaOut) {
        ThrowIfDisposed();
        DropoutForwardState forward = TranslateState<DropoutForwardState>(forwardState);
        DropoutBackwardState backward = TranslateState<DropoutBackwardState>(backwardState);
        
        ElementwiseBackend.Dropout(backward.Delta, forward.Mask, deltaOut);
    }
    
    public override DropoutLayer Copy() {
        ThrowIfDisposed();
        DropoutLayer layer = new DropoutLayer(_dropoutChance, seed);
        layer.Initialize(NeuronCount, Backend, null);
        return layer;
    }

    public override void Save(BinaryWriter writer) {
        ThrowIfDisposed();
        writer.Write("dropout");
        writer.Write(NeuronCount);
        writer.Write(_dropoutChance);
        writer.Write(seed);
    }

    public new static DropoutLayer Load(BinaryReader reader, IBackend backend) {
        int size = reader.ReadInt32();
        float dropoutChance = ValidateChance(reader.ReadSingle());
        int seed = reader.ReadInt32();
        
        if (size <= 0)
            throw new InvalidDataException($"Invalid dropout layer size: {size}.");
        
        DropoutLayer layer = new DropoutLayer(dropoutChance, seed);
        layer.Initialize(size, backend, null);
        
        return layer;
    }

    public override void ValidateLayer(int index, int inputSize) {
        if (NeuronCount != inputSize)
            throw new InvalidDataException(
                $"Dropout layer {index} has {NeuronCount} neurons, expected {inputSize}");
    }

    public override void Dispose() {
        if (Disposed) return;
        Disposed = true;
        
        GC.SuppressFinalize(this);
    }
}