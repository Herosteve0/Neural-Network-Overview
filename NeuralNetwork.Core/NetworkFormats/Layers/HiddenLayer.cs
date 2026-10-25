using NeuralNetwork.Core.Backend;
using NeuralNetwork.Core.Data.SingleValues;
using NeuralNetwork.Core.Data.Tensors;

namespace NeuralNetwork.Core.NetworkFormats.Layers;

public class HiddenLayer : BaseLayer {
    private static float WeightScaler(int previousLength) {
        return (float)Math.Sqrt(6f / previousLength);
    }

    public HiddenLayer(int size, IBackend backend, BaseLayer previousLayer, Random random) : base(size, backend) {
        previousLayer.NextLayer = this;
        Bias = Backend.CreateTensor(0f, size);
        
        float value = WeightScaler(previousLayer.NeuronCount);

        float[] data = new float[size * previousLayer.NeuronCount];
        for (int i = 0; i < data.Length; i++) {
            data[i] = value * (2 * random.NextSingle() - 1);
        }
        
        Weights = Backend.CreateTensor(data, size, previousLayer.NeuronCount);
        WeightsT = Backend.CreateTensor<float>(previousLayer.NeuronCount, size);
        Backend.MatrixTranspose(Weights, WeightsT);
    }
    public HiddenLayer(int size, IBackend backend, BaseLayer previousLayer, float[] weightsValues, float[] biasValues) : base(size, backend) {
        previousLayer.NextLayer = this;
        Bias = Backend.CreateTensor(biasValues, size);
            
        Weights = Backend.CreateTensor(weightsValues, size, previousLayer.NeuronCount);
        WeightsT = Backend.CreateTensor<float>(previousLayer.NeuronCount, size);
        Backend.MatrixTranspose(Weights, WeightsT);
    }

    public readonly ITensor<float> Bias;
    public readonly ITensor<float> Weights;
    public readonly ITensor<float> WeightsT;

    public override void Forward(ITensor<float> input, ITensor<float> valuesOut, ITensor<float> activationOut, ISingleValue<int> predictionOut) {
        Backend.ForwardPass(input, Weights, Bias, valuesOut);
        Backend.Activation(valuesOut, activationOut);
    }

    public override void Backward(ITensor<float> values, ITensor<float> delta, ITensor<float> deltaOut) {
        Backend.Backward(values, delta, NextLayer.WeightsT, deltaOut);
    }

    public override void BackPropagation(ITensor<float> delta, ITensor<float> activations, ITensor<float> weightDelta, ITensor<float> biasDelta) {
        Backend.BackPropagation(delta, activations, weightDelta, biasDelta);
    }

    public override void Adjust(ITensor<float> weightDelta, ITensor<float> biasDelta, float scale) {
        Backend.Adjust(weightDelta, biasDelta, scale, Weights, WeightsT, Bias);
    }

    public override void Dispose() {
        Bias.Dispose();
        Weights.Dispose();
        WeightsT.Dispose();
    }
}