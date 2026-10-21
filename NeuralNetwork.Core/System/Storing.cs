using NeuralNetwork.Core.System.Backend;

namespace NeuralNetwork.Core.System;

public class NetworkFile(string filepath) {
    private string FilePath { get; } = Path.GetFullPath("../../../"  + filepath);

    public void Save(NetworkInstance instance) {
        using BinaryWriter writer = new BinaryWriter(File.Open(FilePath, FileMode.Create));
            
        Network network = instance.Network;

        writer.Write("NeuralNetworkEduProject");
        writer.Write(instance.Accuracy);
            
        writer.Write(network.LayerAmount);
        for (int i = 0; i < network.LayerAmount; i++) {
            writer.Write(network.LayerLength[i]);
        }
            
        (float[][] weights, float[][] biases) = network.GetData();

        foreach (float[] weight in weights) {
            foreach (float value in weight) {
                writer.Write(value);
            }
        }
        foreach (float[] bias in biases) {
            foreach (float value in bias) {
                writer.Write(value);
            }
        }
    }

    public NetworkInstance Load(IBackend backend) {
        if (!Path.Exists(FilePath)) return null;

        using BinaryReader reader = new BinaryReader(File.OpenRead(FilePath));

        string header = reader.ReadString();
        float accuracy = reader.ReadSingle();

        int layerAmount = reader.ReadInt32();
        int[] layers = new int[layerAmount];
        for (int i = 0; i < layerAmount; i++) {
            layers[i] = reader.ReadInt32();
        }
            
        float[][] weights = new float[layerAmount - 1][];
        float[][] bias = new float[layerAmount - 1][];

        for (int i = 0; i < layerAmount - 1; i++) {
            weights[i] = new float[layers[i + 1] * layers[i]];
            for (int j = 0; j < weights[i].Length; j++) {
                weights[i][j] = reader.ReadSingle();
            }
        }

        for (int i = 0; i < layerAmount - 1; i++) {
            bias[i] = new float[layers[i + 1]];
            for (int j = 0; j < bias[i].Length; j++) {
                bias[i][j] = reader.ReadSingle();
            }
        }
            
        Network network = new Network(layers, backend, weights, bias);
        return new NetworkInstance(network, accuracy);
    }

    public float LoadAccuracy() {
        if (!Path.Exists(FilePath)) return 0;

        using BinaryReader reader = new BinaryReader(File.OpenRead(FilePath));

        string header = reader.ReadString();
        return reader.ReadSingle();
            
    }
}