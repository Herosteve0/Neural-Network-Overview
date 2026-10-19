namespace NeuralNetwork.Core.Data;

public readonly record struct DataSample<T>(T[] Values, int Label) {
    public T[] LabelArray(int size, T value) {
        T[] r = new T[size];
        r[Label] = value;
        return r;
    }
}

public record struct DataGuess<T>(DataSample<T> Data, int Prediction) { }

public interface IDataBatch<TSelf> where TSelf : IDataBatch<TSelf> {
    public int Size { get; }
        
    public TSelf GetSmallBatch(int index, int length);
    public void Shuffle(Random random);
}
    
public abstract class BatchBase<TSelf, TItem>(TItem[] data) : IDataBatch<TSelf> where TSelf : BatchBase<TSelf, TItem> {
    public readonly TItem[] Data = data;

    public int Size => Data.Length;
    public TItem this[int index] => Data[index];
    public TItem this[long index] => Data[index];

    protected abstract TSelf CreateBatch(TItem[] data);

    public TSelf GetSmallBatch(int index, int length) {
        TItem[] newData = new TItem[length];
        Array.Copy(Data, index, newData, 0, length);
        return CreateBatch(newData);
    }

    public void Shuffle(Random random) {
        for (int i = Size - 1; i > 0; i--) {
            int r = random.Next(0, i);
            (Data[i], Data[r]) = (Data[r], Data[i]);
        }
    }
}

public class DataBatch<T>(DataSample<T>[] data) : BatchBase<DataBatch<T>, DataSample<T>>(data) {
    protected override DataBatch<T> CreateBatch(DataSample<T>[] data) {
        return new DataBatch<T>(data);
    }
}

public class GuessBatch<T>(DataGuess<T>[] data) : BatchBase<GuessBatch<T>, DataGuess<T>>(data) {
    protected override GuessBatch<T> CreateBatch(DataGuess<T>[] data) {
        return new GuessBatch<T>(data);
    }
}