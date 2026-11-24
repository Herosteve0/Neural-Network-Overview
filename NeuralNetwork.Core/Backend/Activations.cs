using NeuralNetwork.Core.Data.SingleValues;
using NeuralNetwork.Core.Data.Tensors;

namespace NeuralNetwork.Core.Backend;

public abstract class ActivationFunction {
    protected IActivationBackend Backend;
    private bool _initialized;
    
    public void Initialize(IBackend backend) {
        ArgumentNullException.ThrowIfNull(backend);
        if (_initialized)
            throw new InvalidOperationException("The activation function has already been initialized.");
        
        Backend = backend as IActivationBackend
                   ?? throw new NotSupportedException($"The selected backend does not implement activation operations.");
        _initialized = true;
    }

    public abstract void Save(BinaryWriter writer);
    
    public static ActivationFunction Load(BinaryReader reader) {
        string type = reader.ReadString();

        return type switch {
            "relu" => new ReLu(),
            "leaky_relu" => new LeakyReLu(reader.ReadSingle()),
            "softmax" => new Softmax(),
            "sigmoid" => new Sigmoid(),
            _ => throw new InvalidDataException($"Unknown action type '{type}'.")
        };
    }

    public abstract float InitScalar(int inputSize, int outputSize);
    public abstract void Forward(ITensor<float> input, ITensor<float> output, ISingleValue<int> predictionOut);
    public abstract void Backward(ITensor<float> input, ITensor<float> gradient);
}

public sealed class ReLu : ActivationFunction {
    public override void Save(BinaryWriter writer) {
        writer.Write("relu");
    }

    public override float InitScalar(int inputSize, int outputSize) {
        return (float)Math.Sqrt(6f / inputSize);
    }

    public override void Forward(ITensor<float> input, ITensor<float> output, ISingleValue<int> predictionOut) {
        Backend.ReLu_Forward(input, output);
        
        if (predictionOut == null) return;
        ((IBackend)Backend).MaxIndex(output, predictionOut);
    }

    public override void Backward(ITensor<float> input, ITensor<float> gradient) {
        Backend.ReLu_Backward(input, gradient);
    }
}

public sealed class LeakyReLu(float a) : ActivationFunction {
    private readonly float _a = ValidateParameter(a);

    private static float ValidateParameter(float a) {
        return float.IsFinite(a) ? a : throw new ArgumentOutOfRangeException(nameof(a));
    }
    
    public override void Save(BinaryWriter writer) {
        writer.Write("leaky_relu");
        writer.Write(_a);
    }

    public override float InitScalar(int inputSize, int outputSize) {
        return (float)Math.Sqrt(6f / ((1 + _a * _a) * inputSize));
    }

    public override void Forward(ITensor<float> input, ITensor<float> output, ISingleValue<int> predictionOut) {
        Backend.LeakyReLu_Forward(input, _a, output);
        
        if (predictionOut == null) return;
        ((IBackend)Backend).MaxIndex(output, predictionOut);
    }

    public override void Backward(ITensor<float> input, ITensor<float> gradient) {
        Backend.LeakyReLu_Backward(input, _a, gradient);
    }
}

public sealed class Softmax : ActivationFunction {
    public override void Save(BinaryWriter writer) {
        writer.Write("softmax");
    }

    public override float InitScalar(int inputSize, int outputSize) {
        return (float)Math.Sqrt(6f / (inputSize + outputSize));
    }

    public override void Forward(ITensor<float> input, ITensor<float> output, ISingleValue<int> predictionOut) {
        Backend.Softmax_Forward(input, output, predictionOut);
    }

    public override void Backward(ITensor<float> input, ITensor<float> gradient) {
        Backend.Softmax_Backward(input, gradient);
    }
}

public sealed class Sigmoid : ActivationFunction {
    public override void Save(BinaryWriter writer) {
        writer.Write("sigmoid");
    }

    public override float InitScalar(int inputSize, int outputSize) {
        return (float)Math.Sqrt(6f / (inputSize + outputSize));
    }

    public override void Forward(ITensor<float> input, ITensor<float> output, ISingleValue<int> predictionOut) {
        Backend.Sigmoid_Forward(input, output);
        
        if (predictionOut == null) return;
        ((IBackend)Backend).MaxIndex(output, predictionOut);
    }

    public override void Backward(ITensor<float> input, ITensor<float> gradient) {
        Backend.Sigmoid_Backward(input, gradient);
    }
}