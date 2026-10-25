namespace NeuralNetwork.Core.Data.Tensors;

public interface ITensor<T> : IDisposable where T : unmanaged {
    /**
     * <summary>
     * Number of dimensions in this tensor.
     * </summary>
     */
    int Rank { get; }

    /**
     * <summary>
     * The size of each dimension.
     * </summary>
     */
    IReadOnlyList<int> Shape { get; }

    /**
     * <summary>
     * Number of elements in the tensor.
     * </summary>
     */
    int Length { get; }

    /**
     * <summary>
     * Stride of each dimension.
     * Essentially, the offset each dimension needs to go to the index in the array.
     * </summary>
     */
    IReadOnlyList<int> Strides { get; }
}

public abstract class BaseTensor<T> : ITensor<T> where T : unmanaged {
    private readonly int[] _shape;
    private readonly int[] _stride;

    public int Rank => _shape.Length;
    public IReadOnlyList<int> Shape => _shape;
    public IReadOnlyList<int> Strides => _stride;
    public int Length { get; }

    protected BaseTensor(params int[] shape) {
        ValidateShape(shape);

        _shape = (int[])shape.Clone();
        _stride = new int[_shape.Length];

        Length = 1;
        for (int i = _stride.Length - 1; i >= 0; i--) {
            _stride[i] = Length;
            Length *= _shape[i];
        }
    }

    protected int LengthFromShape(params int[] shape) {
        return shape.Aggregate(1, (current, i) => current * i);
    }

    /**
     * <summary>
     * Converts a Nth space point to an index for the data array.
     * </summary>
     */
    protected int GetFlatIndex(params int[] indices) {
        if (indices.Length != Rank)
            throw new ArgumentException($"Expected {Rank} indices, got {indices.Length}.", nameof(indices));

        int index = 0;

        for (int i = 0; i < Rank; i++) {

            int value = indices[i];

            if ((uint)value >= (uint)_shape[i]) {
                throw new IndexOutOfRangeException($"Index {value} is out of dimension {i} with size {_shape[i]}.");
            }

            index += value * _stride[i];
        }

        return index;
    }

    private static void ValidateShape(int[] shape) {
        ArgumentNullException.ThrowIfNull(shape);

        if (shape.Length == 0) {
            throw new ArgumentException("Tensor must have at least one dimension", nameof(shape));
        }

        if (shape.Any(dimension => dimension <= 0)) {
            throw new ArgumentException("Tensor dimension must be greater than zero.", nameof(shape));
        }
    }

    public virtual void Dispose() { }
}
