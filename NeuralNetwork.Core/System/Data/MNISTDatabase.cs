namespace NeuralNetwork.Core.System.Data;

public class MnistDatabase {
    private static int ReadBigEndianInt(BinaryReader br) {
        byte[] bytes = br.ReadBytes(4);
        if (BitConverter.IsLittleEndian) Array.Reverse(bytes);
        return BitConverter.ToInt32(bytes, 0);
    }

    private const string Filepath = "../../../../MNIST/";
        
    public static DataSample<float>[] LoadAllTrainingData() {
        MnistDatabase database = new MnistDatabase(Filepath + "train-images.idx3-ubyte", Filepath + "train-labels.idx1-ubyte");
        return database.ReadBatch(database.Size);
    }

    public static DataSample<float>[] LoadAllTestingData() {
        MnistDatabase database = new MnistDatabase(Filepath + "t10k-images.idx3-ubyte", Filepath + "t10k-labels.idx1-ubyte");
        return database.ReadBatch(database.Size);
    }


    private readonly BinaryReader _brImages;
    private readonly BinaryReader _brLabels;

    public readonly int Size;
    public int Index;
    public readonly int Rows;
    public readonly int Cols;

    public MnistDatabase(string imagePath, string labelPath) {
        _brImages = new BinaryReader(File.OpenRead(imagePath));
        _brLabels = new BinaryReader(File.OpenRead(labelPath));

        int magicImage = ReadBigEndianInt(_brImages); // 2051
        int magicLabels = ReadBigEndianInt(_brLabels); // 2049

        int sizeImage = ReadBigEndianInt(_brImages);
        int sizeLabels = ReadBigEndianInt(_brLabels); ;

        if (sizeImage != sizeLabels) {
            throw new Exception("MNIST Database image and label count not matching! Files might be corrupted.");
        }

        Size = sizeImage;

        Rows = ReadBigEndianInt(_brImages);
        Cols = ReadBigEndianInt(_brImages);
    }

    public void CloseLoad() {
        _brImages.Dispose();
        _brLabels.Dispose();
    }

    public DataSample<float>[] ReadBatch(int batchSize) {
        int loops = Math.Min(batchSize, Size - Index);
        if (loops <= 0) return null;

        DataSample<float>[] r = new DataSample<float>[loops];

        for (int i = 0; i < loops; i++) {
            float[] values = new float[Rows * Cols];
            for (int j = 0; j < values.Length; j++) {
                values[j] = _brImages.ReadByte() / 255f;
            }

            int label = _brLabels.ReadByte();

            r[i] = new DataSample<float>(values, label);
        }

        Index += loops;

        if (Index >= Size) CloseLoad();

        return r;
    }
}