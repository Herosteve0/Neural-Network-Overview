using NeuralNetwork.Core.Backend;
using NeuralNetwork.Core.Data.SingleValues;
using NeuralNetwork.Core.Data.Tensors;
using NeuralNetwork.Core.NetworkFormats.Layers.State;

namespace NeuralNetwork.Core.NetworkFormats.Layers;

public class InputLayer(int size) : BaseLayer(size), ILayer {
    public ILayerState CreatePassState() {
        return new InputLayerState(Backend, NeuronCount);
    }
    
    public void Forward(ITensor<float> input, ILayerState passState, ISingleValue<int> predictionOut, PassMode mode) {
        ThrowIfDisposed();
        InputLayerState state = TranslateState<InputLayerState>(passState);

        Backend.Input(input, state.Output);
    }

    public void Delta(ILayerStateDelta backwardState, ITensor<float> deltaOut) {
        throw new NotSupportedException("Delta is not supported on Input Layers.");
    }
    
    public override InputLayer Copy() {
        InputLayer layer = new InputLayer(NeuronCount);
        layer.Initialize(Backend);
        return layer;
    }
    
    public override void Save(BinaryWriter writer) {
        writer.Write("input");
        writer.Write(NeuronCount);
    }

    public new static InputLayer Load(BinaryReader reader, IBackend backend) {
        int neuronCount = reader.ReadInt32();
        if (neuronCount <= 0)
            throw new InvalidDataException($"Invalid input layer size: {neuronCount}.");
        
        InputLayer layer = new InputLayer(neuronCount);
        layer.Initialize(backend);
        
        return layer;
    }

    public override void Dispose() {
        if (Disposed) return;
        Disposed = true;
        
        GC.SuppressFinalize(this);
    }
}