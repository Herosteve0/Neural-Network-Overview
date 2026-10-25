using NeuralNetwork.Core.Backend;
using NeuralNetwork.Core.Data;
using NeuralNetwork.Core.Data.SingleValues;
using NeuralNetwork.Core.Data.Tensors;

namespace NeuralNetwork.Core.NetworkFormats.Layers;

public class OutputLayer : HiddenLayer {
    public OutputLayer(int size, IBackend backend, BaseLayer previousLayer, Random random) : base(size, backend, previousLayer, random) { }
    public OutputLayer(int size, IBackend backend, BaseLayer previousLayer, float[] weightsValues, float[] biasValues) : base(size, backend, previousLayer, weightsValues, biasValues) { }
        
    public override void Forward(ITensor<float> input, ITensor<float> valuesOut, ITensor<float> activationOut, ISingleValue<int> predictionOut) {
        Backend.ForwardPass(input, Weights, Bias, valuesOut);
        Backend.Output(valuesOut, activationOut, predictionOut);
    }

    public override void Backward(ITensor<float> values, ITensor<float> expectedValues, ITensor<float> deltaOut) {
        Backend.BackwardOutput(values, expectedValues, deltaOut);
    }
}