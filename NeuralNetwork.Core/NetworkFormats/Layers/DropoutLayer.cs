using NeuralNetwork.Core.Backend;
using NeuralNetwork.Core.Data;
using NeuralNetwork.Core.Data.Tensors;

namespace NeuralNetwork.Core.NetworkFormats.Layers;

public class DropoutLayer : HiddenLayer {
    private float _dropoutChance;

    public DropoutLayer(int size, float dropoutChance, IBackend backend, BaseLayer previousLayer, Random random) : base(size, backend, previousLayer, random) {
        _dropoutChance = dropoutChance;
        
        Dropout = Backend.CreateTensor<bool>(size);
    }

    public DropoutLayer(int size, float dropoutChance, IBackend backend, BaseLayer previousLayer, float[] weightsValues, float[] biasValues) : base(size, backend, previousLayer, weightsValues, biasValues) {
        _dropoutChance = dropoutChance;
        
        Dropout = Backend.CreateTensor<bool>(size);
    }
    
    public readonly ITensor<bool> Dropout;
    
    public void ApplyDropout() {
        
    }
    
    public void ResetDropout() {
        // Backend.Fill<bool>(Dropout, false);
    }
}