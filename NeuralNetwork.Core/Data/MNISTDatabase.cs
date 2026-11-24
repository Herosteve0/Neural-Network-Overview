using NeuralNetwork.Core.Others;

namespace NeuralNetwork.Core.Data;

using Dataset = (string images, string labels);

public enum MnistSets {
    Training,
    Testing
}

public class MnistDatabase : IDisposable {
    private static readonly string Filepath = FileManager.WorkspacePath + "MNIST/";
    private static readonly string TrainingImagePath = Filepath + "train-images.idx3-ubyte";
    private static readonly string TrainingLabelsPath = Filepath + "train-labels.idx1-ubyte";
    private static readonly string TestingImagePath = Filepath + "t10k-images.idx3-ubyte";
    private static readonly string TestingLabelsPath = Filepath + "t10k-labels.idx1-ubyte";

    private static Dataset TranslateSet(MnistSets set) {
        return set switch {
            MnistSets.Training => (TrainingImagePath, TrainingLabelsPath),
            MnistSets.Testing => (TestingImagePath, TestingLabelsPath),
            _ => throw new ArgumentOutOfRangeException(nameof(set))
        };
    }
    
    public static DataSample<float>[] LoadAllTrainingData() {
        using MnistDatabase database = new MnistDatabase(MnistSets.Training);
        return database.ReadBatch(database._size);
    }

    public static DataSample<float>[] LoadAllTestingData() {
        using MnistDatabase database = new MnistDatabase(MnistSets.Testing);
        return database.ReadBatch(database._size);
    }

    private readonly BinaryReader _brImages;
    private readonly BinaryReader _brLabels;
    private readonly int _size;
    private readonly int _rows;
    private readonly int _cols;
    private int _index;
    private bool _disposed;

    private MnistDatabase(MnistSets set) {
        BinaryReader images = null;
        BinaryReader labels = null;

        try {
            (string imagePath, string labelPath) = TranslateSet(set);
            images = new BinaryReader(File.OpenRead(imagePath));
            labels = new BinaryReader(File.OpenRead(labelPath));

            int magicImage = ReadBigEndianInt(images); // 2051
            if (magicImage != 2051)
                throw new InvalidDataException($"MNIST image magic number is invalid: {magicImage}.");
            int magicLabels = ReadBigEndianInt(labels); // 2049
            if (magicLabels != 2049)
                throw new InvalidDataException($"MNIST label magic number is invalid: {magicLabels}.");

            int sizeImage = ReadBigEndianInt(images);
            int sizeLabels = ReadBigEndianInt(labels);
            if (sizeImage <= 0 || sizeImage != sizeLabels)
                throw new InvalidDataException("MNIST image and label counts are invalid or do not match.");
            
            int rows = ReadBigEndianInt(images);
            int cols = ReadBigEndianInt(images);
            if (rows <= 0 || cols <= 0)
                throw new InvalidDataException($"Invalid MNIST image dimensions: {rows}x{cols}.");
            
            _size = sizeImage;
            _rows = rows;
            _cols = cols;
            _brImages = images;
            _brLabels = labels;
        } catch {
            labels?.Dispose();
            images?.Dispose();
            throw;
        }
    }
    
    private static int ReadBigEndianInt(BinaryReader br) {
        byte[] bytes = br.ReadBytes(4);
        if (bytes.Length != sizeof(int))
            throw new EndOfStreamException("Unexpected end of MNIST IDX header.");
        if (BitConverter.IsLittleEndian) Array.Reverse(bytes);
        return BitConverter.ToInt32(bytes, 0);
    }

    private DataSample<float>[] ReadBatch(int batchSize) {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentOutOfRangeException.ThrowIfNegative(batchSize);

        int loops = Math.Min(batchSize, _size - _index);
        if (loops <= 0) return [];

        int pixelsPerImage = checked(_rows * _cols);
        DataSample<float>[] result = new DataSample<float>[loops];
        for (int i = 0; i < loops; i++) {
            float[] values = new float[pixelsPerImage];
            for (int j = 0; j < values.Length; j++) {
                values[j] = _brImages.ReadByte() / 255f;
            }

            int label = _brLabels.ReadByte();
            if (label > 9) throw new InvalidDataException($"Invalid MNIST label: {label}.");
            result[i] = new DataSample<float>(values, label);
        }

        _index += loops;
        return result;
    }

    public void Dispose() {
        if (_disposed) return;
        _disposed = true;
        GC.SuppressFinalize(this);
        _brImages.Dispose();
        _brLabels.Dispose();
    }
}