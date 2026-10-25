using NeuralNetwork.Core.Backend;
using NeuralNetwork.Core.Data;
using NeuralNetwork.Core.Data.SingleValues;
using NeuralNetwork.Core.Data.Tensors;

namespace NeuralNetwork.Core.NetworkFormats.Layers;

public abstract class BaseLayer(int size, IBackend backend) : IDisposable {

    protected readonly IBackend Backend = backend;
    public int NeuronCount { get; } = size;
    public HiddenLayer NextLayer { get; protected internal set; }
    
    public abstract void Forward(ITensor<float> input, ITensor<float> valuesOut, ITensor<float> activationOut, ISingleValue<int> predictionOut);
    public abstract void Backward(ITensor<float> values, ITensor<float> delta, ITensor<float> deltaOut);
        
    public abstract void BackPropagation(ITensor<float> delta, ITensor<float> activations, ITensor<float> weightDelta, ITensor<float> biasDelta);
        
    public abstract void Adjust(ITensor<float> weightDelta, ITensor<float> biasDelta, float scale);

    public virtual void Dispose() { }
}