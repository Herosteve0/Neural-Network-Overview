using NeuralNetwork.Core.Backend;
using NeuralNetwork.Core.Data;
using NeuralNetwork.Core.Data.Tensors;
using NeuralNetwork.Core.Iterators;
using NeuralNetwork.Core.NetworkFormats.Layers;

namespace NeuralNetwork.Core.NetworkFormats;

public class GrowableNetwork {
    private NetworkInstance Instance { get; }
    private Network Network => Instance.Network;
    private IBackend Backend => Network.Backend;
    public float Accuracy => Instance.Accuracy;

    public GrowableNetwork(int[] layers, IBackend backend, Random random) : this(
            new Network(layers, backend, random), 0
        ) { }
    public GrowableNetwork(Network network, float accuracy) : this(
        new NetworkInstance(network, accuracy)
    ) { }
    public GrowableNetwork(NetworkInstance instance) {
        Instance = instance;
    }

    public static implicit operator NetworkInstance(GrowableNetwork network) {
        return network.Instance;
    }
    public static implicit operator GrowableNetwork(NetworkInstance instance) {
        return new GrowableNetwork(instance);
    }

    public GrowableNetwork Copy() {
        return new GrowableNetwork(Instance.Copy());
    }

    public void Dispose() {
        Instance.Dispose();
    }

    public async Task<GrowableNetwork> TrainAndEvaluate(
        Trainer trainer, Random random, float learningRate, Tester tester
    ) {
        trainer.Tracker.Reset();
        tester.Tracker.Reset();
        
        await trainer.Train(
            Network,
            random,
            learningRate
        );
        
        float accuracy = await tester.Test(Network);
        
        return new GrowableNetwork(Network, accuracy);
    }

    public GrowableNetwork Mutate(Random random, float mutationStrength) {
        GrowableNetwork newNetwork = Copy();

        if (mutationStrength == 0f) return newNetwork;

        Network network = newNetwork.Network;
        IBackend backend = newNetwork.Backend;

        for (int i = 1; i < network.LayerAmount; i++) {
            HiddenLayer layer = (HiddenLayer)network[i];
				
            ITensor<float> weightsDelta = GenerateOffset(backend, random, layer.Weights, mutationStrength);
            ITensor<float> biasDelta = GenerateOffset(backend, random, layer.Bias, mutationStrength);
				
            backend.Adjust(weightsDelta, biasDelta, 1, layer.Weights, layer.WeightsT, layer.Bias);
            
            weightsDelta.Dispose();
            biasDelta.Dispose();
        }

        return newNetwork;
    }

    private static ITensor<float> GenerateOffset(IBackend backend, Random random, ITensor<float> reference, float mutationStrength) {
        float[] data = new float[reference.Length];

        for (int i = 0; i < data.Length; i++) {
            data[i] = GaussianNoise(random, 0, mutationStrength);
        }
			
        return backend.CreateTensor(data, reference.Shape.ToArray());
    }

    // Temporary
    private static float GaussianNoise(Random random, float mean = 0, float stdDev = 0.1f) {
        float u1 = 1 - random.NextSingle();
        float u2 = 1 - random.NextSingle();

        float randStdNormal = (float)(Math.Sqrt(-2.0f * Math.Log(u1)) * Math.Sin(2 * Math.PI * u2));
			
        return mean + stdDev * randStdNormal;
    }
}