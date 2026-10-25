namespace NeuralNetwork.Core.Data.Tensors;

public class CpuTensor<T> : BaseTensor<T> where T : unmanaged {

    private readonly T[] _data;

    /// <summary>
    /// Direct access to the CPU memory of the tensor.
    ///
    /// Meant to be used by the Backend code.
    /// </summary>
    public Span<T> Data {
        get => _data;
        set {
            if (value.Length != Length) {
                throw new ArgumentException(
                    $"Data contains {value.Length} elements, but the specified shape requires {Length}.",
                    nameof(value));
            }

            value.CopyTo(_data);
        }
    }
    
    public CpuTensor(params int[] shape) : base(shape) {
        _data = new T[Length];
    }

    /// <summary>
    /// Create a tensor from existing data.
    /// Input is copied.
    /// </summary>
    public CpuTensor(T[] data, params int[] shape) : this(shape) {
        if (data.Length != Length) {
            throw new ArgumentException(
                $"Data contains {data.Length} elements, but the specified shape requires {Length}.",
                nameof(data)
            );
        }

        data.CopyTo(_data);
    }

    /// <summary>
    /// Create a tensor filled with a given value.
    /// </summary>
    public CpuTensor(T value, params int[] shape) : this(shape) {
        Array.Fill(_data, value);
    }

    public T this[int index] {
        get => _data[index];
        set => _data[index] = value;
    }
    public T this[int row, int col] {
        get => _data[GetFlatIndex(row, col)];
        set => _data[GetFlatIndex(row, col)] = value;
    }
    public T this[params int[] indices] {
        get => _data[GetFlatIndex(indices)];
        set => _data[GetFlatIndex(indices)] = value;
    }

    /// <summary>
    /// Returns a copy of the data.
    /// </summary>
    public T[] ToArray() {
        return (T[])_data.Clone();
    }
}