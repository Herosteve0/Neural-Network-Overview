using NeuralNetwork.Core.Backend;
using NeuralNetwork.Core.Data;
using NeuralNetwork.Core.Data.SingleValues;
using NeuralNetwork.Core.Data.Tensors;

namespace NeuralNetwork.Core.NetworkFormats.Layers;

public class InputLayer(int size, IBackend backend) : BaseLayer(size, backend) {
    public override void Forward(ITensor<float> input, ITensor<float> valuesOut, ITensor<float> activationOut, ISingleValue<int> predictionOut) {
        Backend.Input(input, activationOut);
    }
        
    public override void Backward(ITensor<float> values, ITensor<float> delta, ITensor<float> deltaOut) {}
    public override void BackPropagation(ITensor<float> delta, ITensor<float> activations, ITensor<float> weightDelta, ITensor<float> biasDelta) { }
    public override void Adjust(ITensor<float> weightDelta, ITensor<float> biasDelta, float scale) { }
}