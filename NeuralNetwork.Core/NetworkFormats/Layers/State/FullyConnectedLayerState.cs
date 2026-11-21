using NeuralNetwork.Core.Backend;
using NeuralNetwork.Core.Data.Tensors;

namespace NeuralNetwork.Core.NetworkFormats.Layers.State;

public class FullyConnectedLayerState : ILayerState, ILayerStateDelta {
    public ITensor<float> Values { get; private set; }
    public ITensor<float> Output { get; private set; }
    public ITensor<float> Delta { get; private set; }
    public ITensor<float> WeightsDelta { get; private set; }
    public ITensor<float> BiasDelta { get; private set; }

    public FullyConnectedLayerState(IBackend backend, int inputSize, int outputSize) {
        try {
            Values = backend.CreateTensor(0f, outputSize);
            Output = backend.CreateTensor(0f, outputSize);
            
            Delta = backend.CreateTensor(0f, outputSize);
            
            WeightsDelta = backend.CreateTensor(0f, outputSize, inputSize);
            BiasDelta = backend.CreateTensor(0f, outputSize);
        } catch {
            Dispose();
            throw;
        }
    }

    public void Dispose() {
        Values?.Dispose();
        Values = null;
        Output?.Dispose();
        Output = null;
        
        Delta?.Dispose();
        Delta = null;
        
        WeightsDelta?.Dispose();
        WeightsDelta = null;
        BiasDelta?.Dispose();
        BiasDelta = null;
        
        GC.SuppressFinalize(this);
    }
}
