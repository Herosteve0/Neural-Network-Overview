namespace NeuralNetwork.Core.System.Data;

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
    
public abstract class BatchBase<TSelf, TItem>(Memory<TItem> data) : IDataBatch<TSelf> where TSelf : BatchBase<TSelf, TItem> {
    public int Size => data.Length;
    public ReadOnlySpan<TItem> DataSpan => data.Span;
    
    public TItem this[int index] => DataSpan[index];
    public TItem this[long index] => DataSpan[(int)index];

    protected abstract TSelf CreateBatch(Memory<TItem> data);

    public TSelf Copy() {
        return CreateBatch(data.ToArray());
    }
    
    public TSelf GetSmallBatch(int index, int length) {
        Memory<TItem> slicedData = data.Slice(index, length);
        return CreateBatch(slicedData);
    }

    public void Shuffle(Random random) {
        Span<TItem> span = data.Span;
        
        for (int i = span.Length - 1; i > 0; i--) {
            int r = random.Next(0, i + 1);
            (span[i], span[r]) = (span[r], span[i]);
        }
    }
}

public class DataBatch<T>(Memory<DataSample<T>> data) : BatchBase<DataBatch<T>, DataSample<T>>(data) {
    protected override DataBatch<T> CreateBatch(Memory<DataSample<T>> data) {
        return new DataBatch<T>(data);
    }
}

public class GuessBatch<T>(Memory<DataGuess<T>> data) : BatchBase<GuessBatch<T>, DataGuess<T>>(data) {
    protected override GuessBatch<T> CreateBatch(Memory<DataGuess<T>> data) {
        return new GuessBatch<T>(data);
    }
}