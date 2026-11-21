using NeuralNetwork.Core.Data.SingleValues;
using NeuralNetwork.Core.Data.Tensors;

namespace NeuralNetwork.Core.Backend;

public interface IActivation {
    void Initialize(IBackend backend);

    void Save(BinaryWriter writer);

    public static IActivation Load(BinaryReader reader) {
        string type = reader.ReadString();

        return type switch {
            "relu" => new ReLu(),
            "leaky_relu" => new LeakyReLu(reader.ReadSingle()),
            "softmax" => new Softmax(),
            "sigmoid" => new Sigmoid(),
            _ => null 
        };
    }
    
    void Forward(ITensor<float> input, ITensor<float> output, ISingleValue<int> predictionOut);
    void Backward(ITensor<float> input, ITensor<float> gradient);
}

public sealed class ReLu : IActivation {
    private IActivationBackend _backend;

    public void Initialize(IBackend backend) {
        _backend = backend as IActivationBackend;
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

    public void Initialize(IBackend backend) {
        _backend = backend as IActivationBackend;
    }

    public void Save(BinaryWriter writer) {
        writer.Write("leaky_relu");
        writer.Write(a);
    }

    public void Forward(ITensor<float> input, ITensor<float> output, ISingleValue<int> predictionOut) {
        _backend.LeakyReLu_Forward(input, a, output);
        
        if (predictionOut == null) return;
        ((IBackend)_backend).MaxIndex(output, predictionOut);
    }

    public void Backward(ITensor<float> input, ITensor<float> gradient) {
        _backend.LeakyReLu_Backward(input, a, gradient);
    }
}

public sealed class Softmax : IActivation {
    private IActivationBackend _backend;

    public void Initialize(IBackend backend) {
        _backend = backend as IActivationBackend;
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
        _backend = backend as IActivationBackend;
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