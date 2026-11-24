using NeuralNetwork.Core.Data.Tensors;

namespace NeuralNetwork.Core.NetworkFormats.Layers.State;

public enum PassMode {
    Training,
    Inference
}

public interface ILayerForwardState : IDisposable {
    public ITensor<float> Output { get; }
}

public interface ILayerBackwardState : IDisposable {
    public ITensor<float> Delta { get; }
}