using NeuralNetwork.Core.Backend;
using NeuralNetwork.Core.Data.SingleValues;
using NeuralNetwork.Core.Data.Tensors;
using NeuralNetwork.Core.NetworkFormats.Layers.State;

namespace NeuralNetwork.Core.NetworkFormats.Layers;

public class FullyConnectedLayer(int size, IActivation activation) : BaseLayer(size), ILayer, ITrainableLayer {
    private static float WeightScaler(int previousLength) {
        return (float)Math.Sqrt(6f / previousLength);
    }
    
    private IMatrixBackend MatrixBackend => Backend as IMatrixBackend 
        ?? throw new NotSupportedException($"The selected backend does not implement matrix operations.");
    private IElementwiseBackend ElementwiseBackend => Backend as IElementwiseBackend
        ?? throw new NotSupportedException("The selected backend does not implement parameter adjustments.");

    private readonly IActivation _activation = activation 
                                               ?? throw new ArgumentNullException(nameof(activation));

    public ITensor<float> Weights { get; private set; }
    public ITensor<float> Bias { get; private set; }

    public override void Initialize(int inputSize, IBackend backend, Random random) {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(NeuronCount);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(inputSize);
        ArgumentNullException.ThrowIfNull(random);
        
        base.Initialize(backend);
        _activation.Initialize(backend);

        try {
            float value = WeightScaler(inputSize);
            float[] data = new float[checked(NeuronCount * inputSize)];
            for (int i = 0; i < data.Length; i++) {
                data[i] = value * (2 * random.NextSingle() - 1);
            }
    
            Weights = Backend.CreateTensor(data, NeuronCount, inputSize);
            Bias = Backend.CreateTensor(0f, NeuronCount);
        } catch {
            Dispose();
            throw;
        }
    }

    public override void InitializeFromData(int inputSize, IBackend backend, float[][] data) {
        ArgumentNullException.ThrowIfNull(data);
        if (data.Length != 2)
            throw new ArgumentException("Fully connected layer data must contain weights and bias.", nameof(data));
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(NeuronCount);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(inputSize);

        if (data[0] is null || data[0].Length != checked(NeuronCount * inputSize))
            throw new ArgumentException("Weight data does not match the requested layer shape.", nameof(data));
        if (data[1] is null || data[1].Length != NeuronCount)
            throw new ArgumentException("Bias data does not match the requested layer shape.", nameof(data));
        
        base.Initialize(backend);
        _activation.Initialize(backend);

        try {
            Weights = Backend.CreateTensor(data[0], NeuronCount, inputSize);
            Bias = Backend.CreateTensor(data[1], NeuronCount);
        } catch {
            Dispose();
            throw;
        }
    }

    public ILayerState CreatePassState() {
        return new FullyConnectedLayerState(Backend, Weights.Shape[1], NeuronCount);
    }

    public void Forward(
        ITensor<float> input, ILayerState passState, ISingleValue<int> predictionOut, PassMode mode
    ) {
        ThrowIfDisposed();
        FullyConnectedLayerState state = TranslateState<FullyConnectedLayerState>(passState);
        
        MatrixBackend.Linear(input, Weights, Bias, state.Values);
        _activation.Forward(state.Values, state.Output, predictionOut);
    }

    public void Delta(ILayerStateDelta backwardState, ITensor<float> deltaOut) {
        ThrowIfDisposed();
        MatrixBackend.Delta(backwardState.Delta, Weights, deltaOut);
    }

    public void ActivationDelta(ILayerStateDelta passState) {
        ThrowIfDisposed();
        FullyConnectedLayerState state = TranslateState<FullyConnectedLayerState>(passState);
        
        _activation.Backward(state.Values, state.Delta);
    }

    public void BackPropagation(
        ITensor<float> activations, ILayerStateDelta passState
    ) {
        ThrowIfDisposed();
        FullyConnectedLayerState state = TranslateState<FullyConnectedLayerState>(passState);
        
        MatrixBackend.BackPropagation(state.Delta, activations, state.WeightsDelta, state.BiasDelta);
    }

    public void Adjust(ILayerState passState, float scale) {
        ThrowIfDisposed();
        FullyConnectedLayerState state = TranslateState<FullyConnectedLayerState>(passState);
        
        ElementwiseBackend.Adjust(state.WeightsDelta, state.BiasDelta, scale, Weights, Bias);
    }
    
    public void Mutate(Random random, float mutationStrength) {
        ThrowIfDisposed();
        if (mutationStrength == 0f) return;
        
        ITensor<float> weightsDelta = null;
        ITensor<float> biasDelta = null;

        try {
            weightsDelta = GenerateOffset(Backend, random, Weights, mutationStrength);
            biasDelta = GenerateOffset(Backend, random, Bias, mutationStrength);
            ElementwiseBackend.Adjust(weightsDelta, biasDelta, 1, Weights, Bias);
        } finally {
            Backend.Synchronize();
            weightsDelta?.Dispose();
            biasDelta?.Dispose();
        }
    }

    private static ITensor<float> GenerateOffset(
        IBackend backend, Random random, ITensor<float> reference, float mutationStrength
    ) {
        float[] data = new float[reference.Length];

        for (int i = 0; i < data.Length; i++) {
            data[i] = GaussianNoise(random, 0, mutationStrength);
        }

        return backend.CreateTensor(data, reference.Shape.ToArray());
    }

    private static float GaussianNoise(Random random, float mean = 0, float stdDev = 0.1f) {
        float u1 = 1 - random.NextSingle();
        float u2 = 1 - random.NextSingle();

        float randStdNormal = (float)(Math.Sqrt(-2.0f * Math.Log(u1)) * Math.Sin(2 * Math.PI * u2));
			
        return mean + stdDev * randStdNormal;
    }

    public override FullyConnectedLayer Copy() {
        ThrowIfDisposed();
        
        FullyConnectedLayer layer = new FullyConnectedLayer(NeuronCount, _activation);
        layer.InitializeFromData(
            Weights.Shape[1],
            Backend,
            [
                Backend.Read(Weights),
                Backend.Read(Bias)
            ]
        );
        return layer;
    }

    public override void Save(BinaryWriter writer) {
        ThrowIfDisposed();
        
        writer.Write("fully_connected");
        writer.Write(NeuronCount);
        writer.Write(Weights.Shape[1]);
        
        _activation.Save(writer);
        
        foreach (float value in Backend.Read(Weights)) {
            writer.Write(value);
        }
        foreach (float value in Backend.Read(Bias)) {
            writer.Write(value);
        }
    }

    public new static FullyConnectedLayer Load(BinaryReader reader, IBackend backend) {
        int neuronCount = reader.ReadInt32();
        int inputSize = reader.ReadInt32();
        
        if (neuronCount <= 0 || inputSize <= 0)
            throw new InvalidDataException($"Invalid fully connected layer shape ({neuronCount}, {inputSize}).");
        
        IActivation activation = IActivation.Load(reader);
        
        FullyConnectedLayer layer = new FullyConnectedLayer(neuronCount, activation);
        
        float[][] data = [new float[neuronCount * inputSize], new float[neuronCount]];

        foreach (float[] d in data) {
            for (int i = 0; i < d.Length; i++) {
                d[i] = reader.ReadSingle();
            }
        }

        layer.InitializeFromData(inputSize, backend, data);
        
        return layer;
    }

    public override void Dispose() {
        if (Disposed) return;
        Disposed = true;
        
        Weights?.Dispose();
        Weights = null;
        
        Bias?.Dispose();
        Bias = null;
        
        GC.SuppressFinalize(this);
    }
}