namespace NeuralNetwork.Core.Data;

public class MnistDatabase {
    private static int ReadBigEndianInt(BinaryReader br) {
        byte[] bytes = br.ReadBytes(4);
        if (BitConverter.IsLittleEndian) Array.Reverse(bytes);
        return BitConverter.ToInt32(bytes, 0);
    }

    private const string Filepath = "../../../../MNIST/";
        
    public static DataSample<float>[] LoadAllTrainingData() {
        MnistDatabase database = new MnistDatabase(Filepath + "train-images.idx3-ubyte", Filepath + "train-labels.idx1-ubyte");
        return database.ReadBatch(database._size);
    }

    public static DataSample<float>[] LoadAllTestingData() {
        MnistDatabase database = new MnistDatabase(Filepath + "t10k-images.idx3-ubyte", Filepath + "t10k-labels.idx1-ubyte");
        return database.ReadBatch(database._size);
    }


    private readonly BinaryReader _brImages;
    private readonly BinaryReader _brLabels;

    private readonly int _size;
    private int _index;
    private readonly int _rows;
    private readonly int _cols;

    private MnistDatabase(string imagePath, string labelPath) {
        _brImages = new BinaryReader(File.OpenRead(imagePath));
        _brLabels = new BinaryReader(File.OpenRead(labelPath));

        int magicImage = ReadBigEndianInt(_brImages); // 2051
        if (magicImage != 2051)
            throw new Exception("MNIST Database image magic number is incorrect!");
        int magicLabels = ReadBigEndianInt(_brLabels); // 2049
        if (magicLabels != 2049)
            throw new Exception("MNIST Database label magic number is incorrect!");

        int sizeImage = ReadBigEndianInt(_brImages);
        int sizeLabels = ReadBigEndianInt(_brLabels);

        if (sizeImage != sizeLabels) {
            throw new Exception("MNIST Database image and label count not matching! Files might be corrupted.");
        }

        _size = sizeImage;

        _rows = ReadBigEndianInt(_brImages);
        _cols = ReadBigEndianInt(_brImages);
    }

    private void CloseLoad() {
        _brImages.Dispose();
        _brLabels.Dispose();
    }

    private DataSample<float>[] ReadBatch(int batchSize) {
        int loops = Math.Min(batchSize, _size - _index);
        if (loops <= 0) return null;

        DataSample<float>[] r = new DataSample<float>[loops];

        for (int i = 0; i < loops; i++) {
            float[] values = new float[_rows * _cols];
            for (int j = 0; j < values.Length; j++) {
                values[j] = _brImages.ReadByte() / 255f;
            }

            int label = _brLabels.ReadByte();

            r[i] = new DataSample<float>(values, label);
        }

        _index += loops;

        if (_index >= _size) CloseLoad();

        return r;
    }
}