using NeuralNetwork.Core.Data.Tensors;

namespace NeuralNetwork.Core.NetworkFormats.Layers.State;

public enum PassMode {
    Training,
    Inference
}

public interface ILayerState : IDisposable {
    public ITensor<float> Output { get; }
}

public interface ILayerStateDelta : IDisposable {
    public ITensor<float> Delta { get; }
}