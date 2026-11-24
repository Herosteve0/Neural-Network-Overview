using NeuralNetwork.Core.Backend;
using NeuralNetwork.Core.NetworkFormats;
using NeuralNetwork.Core.NetworkFormats.Layers;
using NeuralNetwork.Core.NetworkFormats.Networks;

namespace NeuralNetwork.Core.Others;

public class NetworkFile(string filepath) {
    private string FilePath { get; } = FileManager.WorkspacePath + filepath;

    private const string Header = "NeuralNetwork";
    private const string HeaderOld = "NeuralNetworkEduProject";
    private const uint Version = 1;

    public void Save(NetworkInstance instance) {
        ArgumentNullException.ThrowIfNull(instance);
        ArgumentNullException.ThrowIfNull(instance.Network);
        ValidateAccuracy(instance.Accuracy);
        
        string directory = Path.GetDirectoryName(FilePath);
        if (!string.IsNullOrEmpty(directory)) Directory.CreateDirectory(directory);

        string tempPath = FilePath + ".tmp";
        try {
            using (FileStream stream = new FileStream(tempPath, FileMode.Create, FileAccess.Write, FileShare.None)) {
                using (BinaryWriter writer = new BinaryWriter(stream, System.Text.Encoding.UTF8, true)) {
                    writer.Write(Header);
                    writer.Write(Version);
                    writer.Write(instance.Accuracy);
                    instance.Network.Save(writer);
                    writer.Flush();
                    stream.Flush(true);
                }
            }

            File.Move(tempPath, FilePath, true);
        } catch {
            try {
                if (File.Exists(tempPath)) {
                    File.Delete(tempPath);
                }
            } catch { /* Ignored */ }
            throw;
        }
    }

    private static void ValidateAccuracy(float accuracy) {
        if (!float.IsFinite(accuracy) || accuracy is < 0f or > 1f)
            throw new ArgumentOutOfRangeException(nameof(accuracy), "Accuracy must be in range [0, 1].");
    }

    public NetworkInstance Load(IBackend backend, LossFunction lossFunction) {
        if (!Path.Exists(FilePath)) return null;

        using BinaryReader reader = new BinaryReader(File.OpenRead(FilePath));

        string header;
        if ((header = reader.ReadString()) != Header) {
            return header == HeaderOld
                ? ReadOldFormat(reader, backend, lossFunction)
                : throw new InvalidDataException($"Given file contains an invalid file header! {FilePath}");
        }

        uint version;
        if ((version = reader.ReadUInt32()) != Version)
            throw new InvalidDataException($"Unsupported network file version {version}, this build supports version {Version}");
        
        float accuracy = reader.ReadSingle();
        ValidateAccuracy(accuracy);

        Network network = Network.Load(reader, backend, lossFunction);
        
        return new NetworkInstance(network, accuracy);
    }

    public float LoadAccuracy() {
        if (!Path.Exists(FilePath)) return 0;

        using BinaryReader reader = new BinaryReader(File.OpenRead(FilePath));

        string header;
        if ((header = reader.ReadString()) != Header) {
            if (header == HeaderOld) {
                float legacyAccuracy = reader.ReadSingle();
                ValidateAccuracy(legacyAccuracy);
                return legacyAccuracy;
            }
            throw new InvalidDataException($"Given file contains an invalid file header! {FilePath}");
        }

        uint version;
        if ((version = reader.ReadUInt32()) != Version) {
            Console.Error.WriteLine($"Warning: Unsupported network file version {version}, this build supports version {Version}");
        }
        
        float accuracy = reader.ReadSingle();
        ValidateAccuracy(accuracy);
        return accuracy;
    }

    private static NetworkInstance ReadOldFormat(BinaryReader reader, IBackend backend, LossFunction lossFunction) {
        float accuracy = reader.ReadSingle();
        ValidateAccuracy(accuracy);

        int layerAmount = reader.ReadInt32();
        if (layerAmount < 2)
            throw new InvalidDataException($"Invalid legacy network layer count: {layerAmount}.");

        int[] layerLength = new int[layerAmount];
        for (int i = 0; i < layerAmount; i++) {
            layerLength[i] = reader.ReadInt32();
            if (layerLength[i] <= 0)
                throw new InvalidDataException($"Invalid legacy network layer length at index {i}: {layerLength[i]}.");
        }

        float[][] weights = new float[layerAmount - 1][];
        float[][] bias = new float[layerAmount - 1][];

        for (int i = 0; i < layerAmount - 1; i++) {
            weights[i] = new float[layerLength[i + 1] * layerLength[i]];
            for (int j = 0; j < weights[i].Length; j++) {
                weights[i][j] = reader.ReadSingle();
            }
        }

        for (int i = 0; i < layerAmount - 1; i++) {
            bias[i] = new float[layerLength[i + 1]];
            for (int j = 0; j < bias[i].Length; j++) {
                bias[i][j] = reader.ReadSingle();
            }
        }

        BaseLayer[] layers = new BaseLayer[layerAmount];
        int initialized = 0;
        try {
            for (int i = 0; i < layerAmount; i++) {
                int length = layerLength[i];

                if (i == 0) {
                    layers[i] = new InputLayer(length);
                    layers[i].Initialize(backend);
                    initialized++;
                    continue;
                }

                FullyConnectedLayer layer = new FullyConnectedLayer(length,
                    i == layerAmount - 1
                        ? new Softmax()
                        : new ReLu()
                );
                layers[i] = layer;
                layers[i].InitializeFromData(layerLength[i - 1], backend, [weights[i - 1], bias[i - 1]]);
                initialized++;
            }

            Network network = Network.FromLoadedLayers(backend, lossFunction, layers);
            return new NetworkInstance(network, accuracy);
        } catch {
            for (int i = 0; i < initialized; i++) {
                try {
                    layers[i]?.Dispose();
                } catch { /* Ignored */ }
            }
            throw;
        }
    }
}