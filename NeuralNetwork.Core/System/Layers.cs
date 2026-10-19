using NeuralNetwork.Core.Data;

namespace NeuralNetwork.Core;

public abstract class BaseLayer(int size, IBackend backend) : IDisposable {

    protected readonly IBackend Backend = backend;
    public int NeuronCount { get; } = size;
    public Layer NextLayer { get; protected internal set; }
        
    public abstract void Forward(ITensor input, ITensor valuesOut, ITensor activationOut, ISingleValue predictionOut);
    public abstract void Backward(ITensor values, ITensor delta, ITensor deltaOut);
        
    public abstract void BackPropagation(ITensor delta, ITensor activations, ITensor weightDelta, ITensor biasDelta);
        
    public abstract void Adjust(ITensor weightDelta, ITensor biasDelta, float scale);

    public virtual void Dispose() { }
}
    
public class Layer : BaseLayer {
    private static float WeightScaler(int previousLength) {
        return (float)Math.Sqrt(6f / previousLength);
    }

    public Layer(int size, IBackend backend, BaseLayer previousLayer, Random random) : base(size, backend) {
        previousLayer.NextLayer = this;
        Bias = Backend.Zero(size);
            
        float value = WeightScaler(previousLayer.NeuronCount);
        Weights = Backend.Random(random, -value, value, size, previousLayer.NeuronCount);
        WeightsT = Backend.CreateTensor(previousLayer.NeuronCount, size);
        UpdateTranspose();
    }
    public Layer(int size, IBackend backend, BaseLayer previousLayer, float[] weightsValues, float[] biasValues) : base(size, backend) {
        previousLayer.NextLayer = this;
        Bias = Backend.CreateTensor(biasValues, size);
            
        Weights = Backend.CreateTensor(weightsValues, size, previousLayer.NeuronCount);
        WeightsT = Backend.CreateTensor(previousLayer.NeuronCount, size);
        UpdateTranspose();
    }

    public readonly ITensor Bias;
    public readonly ITensor Weights;
    public readonly ITensor WeightsT;

    public void UpdateTranspose() {
        Backend.MatrixTranspose(Weights, WeightsT);
    }

    public override void Forward(ITensor input, ITensor valuesOut, ITensor activationOut, ISingleValue predictionOut) {
        Backend.ForwardPass(input, Weights, Bias, valuesOut);
        Backend.Activation(valuesOut, activationOut);
    }

    public override void Backward(ITensor values, ITensor delta, ITensor deltaOut) {
        Backend.Backward(values, delta, NextLayer.WeightsT, deltaOut);
    }

    public override void BackPropagation(ITensor delta, ITensor activations, ITensor weightDelta, ITensor biasDelta) {
        Backend.BackPropagation(delta, activations, weightDelta, biasDelta);
    }

    public override void Adjust(ITensor weightDelta, ITensor biasDelta, float scale) {
        Backend.Adjust(weightDelta, biasDelta, scale, Weights, WeightsT, Bias);
    }

    public override void Dispose() {
        Bias.Dispose();
        Weights.Dispose();
        WeightsT.Dispose();
    }
}
    
public class InputLayer(int size, IBackend backend) : BaseLayer(size, backend) {
    public override void Forward(ITensor input, ITensor valuesOut, ITensor activationOut, ISingleValue predictionOut) {
        Backend.Input(input, activationOut);
    }
        
    public override void Backward(ITensor values, ITensor delta, ITensor deltaOut) {}
    public override void BackPropagation(ITensor delta, ITensor activations, ITensor weightDelta, ITensor biasDelta) { }
    public override void Adjust(ITensor weightDelta, ITensor biasDelta, float scale) { }
}
    
public class OutputLayer : Layer {
    public OutputLayer(int size, IBackend backend, BaseLayer previousLayer, Random random) : base(size, backend, previousLayer, random) { }
    public OutputLayer(int size, IBackend backend, BaseLayer previousLayer, float[] weightsValues, float[] biasValues) : base(size, backend, previousLayer, weightsValues, biasValues) { }
        
    public override void Forward(ITensor input, ITensor valuesOut, ITensor activationOut, ISingleValue predictionOut) {
        Backend.ForwardPass(input, Weights, Bias, valuesOut);
        Backend.Output(valuesOut, activationOut, predictionOut);
    }

    public override void Backward(ITensor values, ITensor expectedValues, ITensor deltaOut) {
        Backend.BackwardOutput(values, expectedValues, deltaOut);
    }
}