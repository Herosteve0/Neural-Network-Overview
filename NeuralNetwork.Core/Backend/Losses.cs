using NeuralNetwork.Core.Data.SingleValues;
using NeuralNetwork.Core.Data.Tensors;

namespace NeuralNetwork.Core.Backend;

public interface ILoss {
    void Initialize(IBackend backend);

    protected static ILossBackend ReadBackend(IBackend backend) {
        return backend as ILossBackend
               ?? throw new NotSupportedException($"The selected backend does not implement loss operations.");
    }
    
    void Calculate(ITensor<float> activation, ISingleValue<int> label, ISingleValue<float> lossOut);
    void Backward(ITensor<float> input, ITensor<float> expectedInput, ITensor<float> output);
}

public sealed class MeanSquaredError(int a = 1) : ILoss {
    private ILossBackend _backend;

    public void Initialize(IBackend backend) {
        _backend = ILoss.ReadBackend(backend);
    }

    public void Calculate(ITensor<float> activation, ISingleValue<int> label, ISingleValue<float> lossOut) {
        _backend.MSE_Calculate(activation, label, a, lossOut);
    }

    public void Backward(ITensor<float> input, ITensor<float> expectedInput, ITensor<float> output) {
        _backend.MSE_Backward(input, expectedInput, a, output);
    }
}

public sealed class CrossEntropy : ILoss {
    private ILossBackend _backend;

    public void Initialize(IBackend backend) {
        _backend = ILoss.ReadBackend(backend);
    }

    public void Calculate(ITensor<float> activation, ISingleValue<int> label, ISingleValue<float> lossOut) {
        _backend.CrossEntropy_Calculate(activation, label, lossOut);
    }

    public void Backward(ITensor<float> input, ITensor<float> expectedInput, ITensor<float> output) {
        _backend.CrossEntropy_Backward(input, expectedInput, output);
    }
}