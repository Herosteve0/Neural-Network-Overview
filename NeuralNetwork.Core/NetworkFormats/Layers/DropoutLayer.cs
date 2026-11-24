using NeuralNetwork.Core.Backend;
using NeuralNetwork.Core.Data.SingleValues;
using NeuralNetwork.Core.Data.Tensors;
using NeuralNetwork.Core.NetworkFormats.Layers.State;

namespace NeuralNetwork.Core.NetworkFormats.Layers;

public class DropoutLayer(float dropoutChance, int seed) : BaseLayer(0), ILayer {
    private readonly float _dropoutChance = ValidateChance(dropoutChance);
    private readonly int _seed = seed;

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

    public ILayerState CreatePassState() {
        return new DropoutLayerState(Backend, NeuronCount, _seed);
    }

    public void Forward(
        ITensor<float> input, ILayerState passState, ISingleValue<int> predictionOut, PassMode mode
    ) {
        ThrowIfDisposed();
        DropoutLayerState state = TranslateState<DropoutLayerState>(passState);

        if (mode == PassMode.Inference) {
            Backend.Copy(input, state.Output);
            return;
        }
        
        RandomBackend.GenerateDropoutMask(state.Random, _dropoutChance, state.Mask);
        ElementwiseBackend.Dropout(input, state.Mask, state.Output);
    }

    public void Delta(ILayerStateDelta passState, ITensor<float> deltaOut) {
        ThrowIfDisposed();
        DropoutLayerState state = TranslateState<DropoutLayerState>(passState);
        
        ElementwiseBackend.Dropout(state.Delta, state.Mask, deltaOut);
    }
    
    public override DropoutLayer Copy() {
        ThrowIfDisposed();
        DropoutLayer layer = new DropoutLayer(_dropoutChance, _seed);
        layer.Initialize(NeuronCount, Backend, null);
        return layer;
    }

    public override void Save(BinaryWriter writer) {
        ThrowIfDisposed();
        writer.Write("dropout");
        writer.Write(NeuronCount);
        writer.Write(_dropoutChance);
        writer.Write(_seed);
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