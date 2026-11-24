using NeuralNetwork.Core.Backend;
using NeuralNetwork.Core.Data.SingleValues;
using NeuralNetwork.Core.Data.Tensors;
using NeuralNetwork.Core.NetworkFormats.Layers.State;

namespace NeuralNetwork.Core.NetworkFormats.Layers;

public class FullyConnectedLayer(int size, ActivationFunction activation) : BaseLayer(size), ILayer, ITrainableLayer {
    private IMatrixBackend MatrixBackend => Backend as IMatrixBackend 
        ?? throw new NotSupportedException($"The selected backend does not implement matrix operations.");
    private IElementwiseBackend ElementwiseBackend => Backend as IElementwiseBackend
        ?? throw new NotSupportedException("The selected backend does not implement parameter adjustments.");

    private readonly ActivationFunction _activation = activation 
                                                      ?? throw new ArgumentNullException(nameof(activation));

    private ITensor<float> _weights;
    private ITensor<float> _bias;

    public override void Initialize(int inputSize, IBackend backend, Random random) {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(NeuronCount);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(inputSize);
        ArgumentNullException.ThrowIfNull(random);
        
        base.Initialize(backend);
        _activation.Initialize(backend);

        try {
            float value = _activation.InitScalar(inputSize, NeuronCount);
            float[] data = new float[checked(NeuronCount * inputSize)];
            for (int i = 0; i < data.Length; i++) {
                data[i] = value * (2 * random.NextSingle() - 1);
            }
    
            _weights = Backend.CreateTensor(data, NeuronCount, inputSize);
            _bias = Backend.CreateTensor(0f, NeuronCount);
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
            _weights = Backend.CreateTensor(data[0], NeuronCount, inputSize);
            _bias = Backend.CreateTensor(data[1], NeuronCount);
        } catch {
            Dispose();
            throw;
        }
    }

    public ILayerForwardState CreateForwardState() {
        return new FullyConnectedForwardState(Backend, NeuronCount);
    }
    public ILayerBackwardState CreateBackwardState() {
        return new FullyConnectedBackwardState(Backend, _weights.Shape[1], NeuronCount);
    }

    public void Forward(
        ITensor<float> input, ILayerForwardState passState, ISingleValue<int> predictionOut, PassMode mode
    ) {
        ThrowIfDisposed();
        FullyConnectedForwardState state = TranslateState<FullyConnectedForwardState>(passState);
        
        MatrixBackend.Linear(input, _weights, _bias, state.Values);
        _activation.Forward(state.Values, state.Output, predictionOut);
    }

    public void Delta(ILayerForwardState forwardState, ILayerBackwardState backwardState, ITensor<float> deltaOut) {
        ThrowIfDisposed();
        
        MatrixBackend.Delta(backwardState.Delta, _weights, deltaOut);
    }

    public void ActivationDelta(ILayerForwardState forwardState, ILayerBackwardState backwardState) {
        ThrowIfDisposed();
        FullyConnectedForwardState forward = TranslateState<FullyConnectedForwardState>(forwardState);
        
        _activation.Backward(forward.Values, backwardState.Delta);
    }

    public void BackPropagation(
        ITensor<float> activations, ILayerBackwardState backwardState
    ) {
        ThrowIfDisposed();
        FullyConnectedBackwardState backward = TranslateState<FullyConnectedBackwardState>(backwardState);
        
        MatrixBackend.BackPropagation(backward.Delta, activations, backward.WeightsDelta, backward.BiasDelta);
    }

    public void Adjust(ILayerBackwardState backwardState, float scale) {
        ThrowIfDisposed();
        FullyConnectedBackwardState backward = TranslateState<FullyConnectedBackwardState>(backwardState);
        
        ElementwiseBackend.Adjust(backward.WeightsDelta, backward.BiasDelta, scale, _weights, _bias);
    }
    
    public void Mutate(Random random, float mutationStrength) {
        ThrowIfDisposed();
        if (mutationStrength == 0f) return;
        
        ITensor<float> weightsDelta = null;
        ITensor<float> biasDelta = null;

        try {
            weightsDelta = GenerateOffset(Backend, random, _weights, mutationStrength);
            biasDelta = GenerateOffset(Backend, random, _bias, mutationStrength);
            ElementwiseBackend.Adjust(weightsDelta, biasDelta, 1, _weights, _bias);
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
            _weights.Shape[1],
            Backend,
            [
                Backend.Read(_weights),
                Backend.Read(_bias)
            ]
        );
        return layer;
    }

    public override void Save(BinaryWriter writer) {
        ThrowIfDisposed();
        
        writer.Write("fully_connected");
        writer.Write(NeuronCount);
        writer.Write(_weights.Shape[1]);
        
        _activation.Save(writer);
        
        foreach (float value in Backend.Read(_weights)) {
            writer.Write(value);
        }
        foreach (float value in Backend.Read(_bias)) {
            writer.Write(value);
        }
    }

    public new static FullyConnectedLayer Load(BinaryReader reader, IBackend backend) {
        int neuronCount = reader.ReadInt32();
        int inputSize = reader.ReadInt32();
        
        if (neuronCount <= 0 || inputSize <= 0)
            throw new InvalidDataException($"Invalid fully connected layer shape ({neuronCount}, {inputSize}).");
        
        ActivationFunction activation = ActivationFunction.Load(reader);
        
        FullyConnectedLayer layer = new FullyConnectedLayer(neuronCount, activation);
        
        float[][] data = [
            new float[checked(neuronCount * inputSize)],
            new float[neuronCount]
        ];

        foreach (float[] d in data) {
            for (int i = 0; i < d.Length; i++) {
                d[i] = reader.ReadSingle();
            }
        }

        layer.InitializeFromData(inputSize, backend, data);
        
        return layer;
    }

    public float[][] GetWeights() {
        ThrowIfDisposed();

        float[] data = Backend.Read(_weights);
        float[][] r = new float[_weights.Shape[0]][];

        for (int i = 0; i < _weights.Shape[0]; i++) {
            r[i] = new float[_weights.Shape[1]];
            for (int j = 0; j < _weights.Shape[1]; j++) {
                r[i][j] = data[i * _weights.Shape[1] + j];
            }
        }
        
        return r;
    }

    public override void ValidateLayer(int index, int inputSize) {
    if (_weights is null || _bias is null ||
        _weights.Rank != 2 ||
        _weights.Shape[0] != NeuronCount ||
        _weights.Shape[1] != inputSize ||
        _bias.Length != NeuronCount)
        throw new InvalidDataException($"Fully connected layer {index} has inconsistent parameter shapes.");
    }

    public override void Dispose() {
        if (Disposed) return;
        
        _weights?.Dispose();
        _weights = null;
        
        _bias?.Dispose();
        _bias = null;
        
        Disposed = true;
        GC.SuppressFinalize(this);
    }
}