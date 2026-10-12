namespace Neurtal_Network_Overview.System.Data;

public class MnistDatabase {
    private static int ReadBigEndianInt(BinaryReader br) {
        byte[] bytes = br.ReadBytes(4);
        if (BitConverter.IsLittleEndian) Array.Reverse(bytes);
        return BitConverter.ToInt32(bytes, 0);
    }

    private const string Filepath = "../../../MNIST/";
        
    public static DataSample<float>[] LoadAllTrainingData() {
        Console.WriteLine(Path.GetFullPath(Filepath));
        MnistDatabase database = new MnistDatabase(Filepath + "train-images.idx3-ubyte", Filepath + "train-labels.idx1-ubyte");
        return database.ReadBatch(database.Size);
    }

    public static DataSample<float>[] LoadAllTestingData() {
        MnistDatabase database = new MnistDatabase(Filepath + "t10k-images.idx3-ubyte", Filepath + "t10k-labels.idx1-ubyte");
        return database.ReadBatch(database.Size);
    }


    readonly BinaryReader br_images;
    readonly BinaryReader br_labels;

    public readonly int Size;
    public int Index;
    public readonly int Rows;
    public readonly int Cols;

    public MnistDatabase(string imagePath, string labelPath) {
        br_images = new BinaryReader(File.OpenRead(imagePath));
        br_labels = new BinaryReader(File.OpenRead(labelPath));

        int magicImage = ReadBigEndianInt(br_images); // 2051
        int magicLabels = ReadBigEndianInt(br_labels); // 2049

        int sizeImage = ReadBigEndianInt(br_images);
        int sizeLabels = ReadBigEndianInt(br_labels); ;

        if (sizeImage != sizeLabels) {
            throw new Exception("MNIST Database image and label count not matching! Files might be corrupted.");
        }

        Size = sizeImage;

        Rows = ReadBigEndianInt(br_images);
        Cols = ReadBigEndianInt(br_images);
    }

    public void CloseLoad() {
        br_images.Dispose();
        br_labels.Dispose();
    }

    public DataSample<float>[] ReadBatch(int batchSize) {
        int loops = Math.Min(batchSize, Size - Index);
        if (loops <= 0) return null;

        DataSample<float>[] r = new DataSample<float>[loops];

        for (int i = 0; i < loops; i++) {
            float[] values = new float[Rows * Cols];
            for (int j = 0; j < values.Length; j++) {
                values[j] = br_images.ReadByte() / 255f;
            }

            int label = br_labels.ReadByte();

            r[i] = new DataSample<float>(values, label);
        }

        Index += loops;

        if (Index >= Size) CloseLoad();

        return r;
    }
}