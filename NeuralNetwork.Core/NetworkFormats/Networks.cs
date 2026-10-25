using NeuralNetwork.Core.Backend;
using NeuralNetwork.Core.Data;
using NeuralNetwork.Core.NetworkFormats.Layers;

namespace NeuralNetwork.Core.NetworkFormats;

using NetworkValues = (float[][] weights, float[][] bias);

/**
 * <param name="Accuracy">
 * The accuracy of the network. In range [0, 1].
 * </param>
 */
public record NetworkInstance(Network Network, float Accuracy) {
    public NetworkInstance Copy() {
        return this with { Network = Network.Copy() };
    }
    
    public static implicit operator Network(NetworkInstance instance) {
        return instance.Network;
    }
    
    public void Dispose() {
        Network.Dispose();
    }
}
    
public class Network : IDisposable{
    private BaseLayer[] Layers { get; }
    public IBackend Backend { get; }
        
    public Network(int[] layers, IBackend backend, Random random) {
        ArgumentNullException.ThrowIfNull(backend);
        if (layers.Length == 0)
            throw new ArgumentException("Network size must have at least one layer.", nameof(layers));
        if (layers.Any(layer => layer == 0))
            throw new ArgumentException("Layers cannot have length 0.", nameof(layers));
        
        Backend = backend;
        LayerAmount = layers.Length;
        LayerLength = layers.ToArray();
            
        Layers = new BaseLayer[LayerAmount];
        for (int i = 0; i < LayerAmount; i++) {
            int length = layers[i];
            Layers[i] = i == LayerAmount - 1
                ? new OutputLayer(length, backend, Layers[i - 1], random)
                : i == 0
                    ? new InputLayer(length, backend)
                    : new HiddenLayer(length, backend, Layers[i - 1], random);
        }
    }
    public Network(int[] layers, IBackend backend, float[][] weights, float[][] bias) {
        Backend = backend;
        LayerAmount = layers.Length;
        LayerLength = layers.ToArray();
            
        Layers = new BaseLayer[LayerAmount];
        for (int i = 0; i < LayerAmount; i++) {
            int length = layers[i];
            Layers[i] = i == LayerAmount - 1
                ? new OutputLayer(length, backend, Layers[i - 1], weights[i - 1], bias[i - 1])
                : i == 0
                    ? new InputLayer(length, backend)
                    : new HiddenLayer(length, backend, Layers[i - 1], weights[i - 1], bias[i - 1]);
        }
    }
        
    public BaseLayer this[int index] => Layers[index];


    public int LayerAmount { get; }
    public int[] LayerLength { get; }

    public DataGuess<float> QuickCalculate(DataSample<float> input) {
        VirtualNetwork network = new VirtualNetwork(this);
        
        network.LoadBuffer(new DataBatch<float>(input));
        network.CompleteForwardPass(this);
        
        DataGuess<float> result = network.RawGuess(0);
        network.Dispose();
        return result;
    }
        
    public NetworkValues GetData() {
        float[][] weights = new float[LayerAmount - 1][];
        float[][] bias = new float[LayerAmount - 1][];
            
        for (int i = 1; i < LayerAmount; i++) {
            weights[i - 1] = new float[LayerLength[i] * LayerLength[i - 1]];
            Backend.Read(((HiddenLayer)this[i]).Weights).CopyTo(weights[i - 1]);
                
            bias[i - 1] = new float[LayerLength[i]];
            Backend.Read(((HiddenLayer)this[i]).Bias).CopyTo(bias[i - 1]);
        }
            
        return (weights, bias);
    }
    public Network Copy() {
        NetworkValues variables = GetData();
        return new Network(LayerLength, Backend, variables.weights, variables.bias);
    }

    public void Dispose() {
        Backend.Synchronize();
        foreach (BaseLayer layer in Layers) {
            layer.Dispose();
        }
    }
}