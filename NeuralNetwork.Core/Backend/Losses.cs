using NeuralNetwork.Core.Data.SingleValues;
using NeuralNetwork.Core.Data.Tensors;

namespace NeuralNetwork.Core.Backend;

public abstract class LossFunction {
    protected ILossBackend Backend;
    private bool _initialized;

    public void Initialize(IBackend backend) {
        if (_initialized)
            throw new InvalidOperationException("The loss function has already been initialized.");
        
        Backend = backend as ILossBackend
               ?? throw new NotSupportedException($"The selected backend does not implement loss operations.");
        _initialized = true;
    }
    
    /**
     * Calculates the loss of the given activation output.
     */
    public abstract void Calculate(ITensor<float> activation, ISingleValue<int> label, ISingleValue<float> lossOut);
    /**
     * Initializes the first gradient using the loss function.
     */
    public abstract void Backward(ITensor<float> input, ITensor<float> expectedInput, ITensor<float> output);
}

public sealed class MeanSquaredError(float a = 1) : LossFunction {
    private readonly float _a = ValidateParameter(a);

    private static float ValidateParameter(float a) {
        return float.IsFinite(a) ? a : throw new ArgumentOutOfRangeException(nameof(a));
    }
    
    public override void Calculate(ITensor<float> activation, ISingleValue<int> label, ISingleValue<float> lossOut) {
        Backend.MSE_Calculate(activation, label, _a, lossOut);
    }

    public override void Backward(ITensor<float> input, ITensor<float> expectedInput, ITensor<float> output) {
        Backend.MSE_Backward(input, expectedInput, _a, output);
    }
}

public sealed class CrossEntropy : LossFunction {
    public override void Calculate(ITensor<float> activation, ISingleValue<int> label, ISingleValue<float> lossOut) {
        Backend.CrossEntropy_Calculate(activation, label, lossOut);
    }

    public override void Backward(ITensor<float> input, ITensor<float> expectedInput, ITensor<float> output) {
        Backend.CrossEntropy_Backward(input, expectedInput, output);
    }
}