using NeuralNetwork.Core.Data.SingleValues;
using NeuralNetwork.Core.Data.Tensors;

namespace NeuralNetwork.Core.Backend;

public interface IActivation {
    void Initialize(IBackend backend);

    void Save(BinaryWriter writer);

    protected static IActivationBackend ReadBackend(IBackend backend) {
        return backend as IActivationBackend
            ?? throw new NotSupportedException($"The selected backend does not implement activation operations.");
    }
    
    public static IActivation Load(BinaryReader reader) {
        string type = reader.ReadString();

        return type switch {
            "relu" => new ReLu(),
            "leaky_relu" => new LeakyReLu(reader.ReadSingle()),
            "softmax" => new Softmax(),
            "sigmoid" => new Sigmoid(),
            _ => throw new InvalidDataException($"Unknown action type '{type}'.")
        };
    }
    
    void Forward(ITensor<float> input, ITensor<float> output, ISingleValue<int> predictionOut);
    void Backward(ITensor<float> input, ITensor<float> gradient);
}

public sealed class ReLu : IActivation {
    private IActivationBackend _backend;

    public void Initialize(IBackend backend) {
        _backend = IActivation.ReadBackend(backend);
    }

    public void Save(BinaryWriter writer) {
        writer.Write("relu");
    }

    public void Forward(ITensor<float> input, ITensor<float> output, ISingleValue<int> predictionOut) {
        _backend.ReLu_Forward(input, output);
        
        if (predictionOut == null) return;
        ((IBackend)_backend).MaxIndex(output, predictionOut);
    }

    public void Backward(ITensor<float> input, ITensor<float> gradient) {
        _backend.ReLu_Backward(input, gradient);
    }
}

public sealed class LeakyReLu(float a) : IActivation {
    private IActivationBackend _backend;
    private readonly float _a = ValidateParameter(a);

    private static float ValidateParameter(float a) {
        if (!float.IsFinite(a))
            throw new ArgumentOutOfRangeException(nameof(a));
        return a;
    }
    
    public void Initialize(IBackend backend) {
        _backend = IActivation.ReadBackend(backend);
    }

    public void Save(BinaryWriter writer) {
        writer.Write("leaky_relu");
        writer.Write(_a);
    }

    public void Forward(ITensor<float> input, ITensor<float> output, ISingleValue<int> predictionOut) {
        _backend.LeakyReLu_Forward(input, _a, output);
        
        if (predictionOut == null) return;
        ((IBackend)_backend).MaxIndex(output, predictionOut);
    }

    public void Backward(ITensor<float> input, ITensor<float> gradient) {
        _backend.LeakyReLu_Backward(input, _a, gradient);
    }
}

public sealed class Softmax : IActivation {
    private IActivationBackend _backend;

    public void Initialize(IBackend backend) {
        _backend = IActivation.ReadBackend(backend);
    }

    public void Save(BinaryWriter writer) {
        writer.Write("softmax");
    }

    public void Forward(ITensor<float> input, ITensor<float> output, ISingleValue<int> predictionOut) {
        _backend.Softmax_Forward(input, output, predictionOut);
    }

    public void Backward(ITensor<float> input, ITensor<float> gradient) {
        _backend.Softmax_Backward(input, gradient);
    }
}

public sealed class Sigmoid : IActivation {
    private IActivationBackend _backend;

    public void Initialize(IBackend backend) {
        _backend = IActivation.ReadBackend(backend);
    }

    public void Save(BinaryWriter writer) {
        writer.Write("sigmoid");
    }

    public void Forward(ITensor<float> input, ITensor<float> output, ISingleValue<int> predictionOut) {
        _backend.Sigmoid_Forward(input, output);
        
        if (predictionOut == null) return;
        ((IBackend)_backend).MaxIndex(output, predictionOut);
    }

    public void Backward(ITensor<float> input, ITensor<float> gradient) {
        _backend.Sigmoid_Backward(input, gradient);
    }
}