using NeuralNetwork.Core.Backend;
using NeuralNetwork.Core.Data.Tensors;

namespace NeuralNetwork.Core.NetworkFormats.Layers.State;

public class InputLayerState : ILayerState {
    public ITensor<float> Output { get; private set; }

    public InputLayerState(IBackend backend, int outputSize) {
        try {
            Output = backend.CreateTensor(0f, outputSize);
        } catch {
            Dispose();
            throw;
        }
    }

    public void Dispose() {
        Output?.Dispose();
        Output = null;
        
        GC.SuppressFinalize(this);
    }
}
