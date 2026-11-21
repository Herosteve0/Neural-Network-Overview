using NeuralNetwork.Core.Backend;
using NeuralNetwork.Core.Data.Tensors;

namespace NeuralNetwork.Core.NetworkFormats.Layers.State;

public class DropoutLayerState : ILayerState, ILayerStateDelta {
    public ITensor<float> Mask { get; private set; }
    public Random Random { get; private set; }
    public ITensor<float> Output { get; private set; }
    public ITensor<float> Delta { get; private set; }

    public DropoutLayerState(IBackend backend, int outputSize, int seed) {
        try {
            Mask = backend.CreateTensor(0f, outputSize);
            Random = new Random(seed);
            Output = backend.CreateTensor(0f, outputSize);
            Delta = backend.CreateTensor(0f, outputSize);
        } catch {
            Dispose();
            throw;
        }
    }
    
    public void Dispose() {
        Mask?.Dispose();
        Mask = null;
        
        Output?.Dispose();
        Output = null;
        
        Delta?.Dispose();
        Delta = null;
        
        GC.SuppressFinalize(this);
    }
}