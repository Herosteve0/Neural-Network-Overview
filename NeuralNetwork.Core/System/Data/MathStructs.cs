using ILGPU;
using ILGPU.Runtime;

namespace NeuralNetwork.Core.System.Data;

public interface ITensor : IDisposable {
    /// <summary>
    /// Number of dimensions in this tensor.
    /// </summary>
    int Rank { get; }

    /// <summary>
    /// The size of each dimension.
    /// </summary>
    IReadOnlyList<int> Shape { get; }

    /// <summary>
    /// Number of elements in the tensor.
    /// </summary>
    int Length { get; }

    /// <summary>
    /// Stride of each dimension.
    ///
    /// Essentially, how many indices should I add in order to move N in that dimension.
    /// </summary>
    IReadOnlyList<int> Strides { get; }

    /// <summary>
    /// Converts a Nth space point to an index for the data array.
    /// </summary>
    int GetFlatIndex(params int[] indices);
}

public abstract class BaseTensor : ITensor {

    private readonly int[] _shape;
    private readonly int[] _stride;

    public int Rank => _shape.Length;
    public IReadOnlyList<int> Shape => _shape;
    public IReadOnlyList<int> Strides => _stride;
    public int Length { get; }

    public BaseTensor(params int[] shape) {
        ValidateShape(shape);

        _shape = (int[])shape.Clone();
        _stride = new int[_shape.Length];

        Length = 1;
        for (int i = _stride.Length - 1; i >= 0; i--) {
            _stride[i] = Length;
            Length *= _shape[i];
        }
    }

    public int GetFlatIndex(params int[] indices) {
        if (indices.Length != Rank)
            throw new ArgumentException($"Expected {Rank} indicies, got {indices.Length}.", nameof(indices));

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
        if (shape is null) throw new ArgumentNullException(nameof(shape));

        if (shape.Length == 0) {
            throw new ArgumentException("Tensor must have at least one dimension", nameof(shape));
        }

        foreach (int dimension in shape) {
            if (dimension <= 0)
                throw new ArgumentException("Tensor dimension must be greater than zero.", nameof(shape));
        }
    }

    public virtual void Dispose() {
    }
}

public class CpuTensor : BaseTensor {

    private readonly float[] _data;

    /// <summary>
    /// Direct access to the CPU memory of the tensor.
    ///
    /// Meant to be used by the Backend code.
    /// </summary>
    public Span<float> Data {
        get => _data;
        set {
            if (value.Length != Length) {
                throw new ArgumentException(
                    $"Data contains {value.Length} elements, but the specified shape requires {Length}.",
                    nameof(value));
            }

            Array.Copy(value.ToArray(), _data, _data.Length);
        }
    }

    public CpuTensor(params int[] shape) : base(shape) {
        _data = new float[Length];
    }

    /// <summary>
    /// Create a tensor from existing data.
    /// Input is copied.
    /// </summary>
    public CpuTensor(float[] data, params int[] shape) : this(shape) {
        if (data.Length != Length) {
            throw new ArgumentException(
                $"Data contains {data.Length} elements, but the specified shape requires {Length}.",
                nameof(data)
            );
        }

        Array.Copy(data, _data, Length);
    }

    public float this[params int[] indices] {
        get => _data[GetFlatIndex(indices)];
        set => _data[GetFlatIndex(indices)] = value;
    }

    /// <summary>
    /// Returns a copy of the data.
    /// </summary>
    public float[] ToArray() {
        return (float[])_data.Clone();
    }

    /// <summary>
    /// Fills the tensor with a single value.
    /// </summary>
    public void Fill(float value) {
        Array.Fill(_data, value);
    }

    /// <summary>
    /// Creates a tensor filled with a given value.
    /// </summary>
    public static CpuTensor SingleValue(float value, params int[] shape) {
        CpuTensor tensor = new CpuTensor(shape);
        tensor.Fill(value);
        return tensor;
    }

    /// <summary>
    /// Creates a tensor filled with zeros.
    /// </summary>
    public static CpuTensor Zero(params int[] shape) {
        return SingleValue(0, shape);
    }

    /// <summary>
    /// Creates a tensor filled with ones.
    /// </summary>
    public static CpuTensor One(params int[] shape) {
        return SingleValue(1, shape);
    }

    /// <summary>
    /// Creates a randomized tensor.
    /// </summary>
    public static CpuTensor Random(Random random, float min, float max, params int[] shape) {
        CpuTensor tensor = new CpuTensor(shape);
        for (int i = 0; i < tensor.Length; i++) {
            tensor._data[i] = min + (max - min) * random.NextSingle();
        }

        return tensor;
    }

    /// <summary>
    /// Creates a randomized tensor along with it's tranpose.
    /// </summary>
    public static (CpuTensor, CpuTensor) RandomMatrixWithTranspose(Random random, float min, float max,
        (int rows, int columns) shape) {
        CpuTensor tensor = new CpuTensor(shape.rows, shape.columns);
        CpuTensor tensorT = new CpuTensor(shape.columns, shape.rows);

        for (int i = 0; i < tensor.Shape[0]; i++) {
            for (int j = 0; j < tensor.Shape[1]; j++) {
                tensor[i, j] = min + (max - min) * random.NextSingle();
                tensorT[j, i] = tensor[i, j];
            }
        }

        return (tensor, tensorT);
    }
}

public class GpuTensor : BaseTensor {

    private bool _disposed;

    /// <summary>
    /// The owning GPU buffer.
    ///
    /// Meant to be used for Backend code.
    /// </summary>
    public MemoryBuffer1D<float, Stride1D.Dense> Buffer { get; }

    /// <summary>
    /// A non-owning vew of the GPU memory.
    /// </summary>
    public ArrayView1D<float, Stride1D.Dense> View => Buffer.View;

    public GpuTensor(Accelerator accelerator, params int[] shape) : base(shape) {
        ArgumentNullException.ThrowIfNull(accelerator);

        Buffer = accelerator.Allocate1D<float>(Length);
    }

    /// <summary>
    /// Create a GPU tensor and initializes it with CPU data.
    /// </summary>
    public GpuTensor(Accelerator accelerator, float[] data, params int[] shape) : this(accelerator, shape) {
        ArgumentNullException.ThrowIfNull(data);

        if (data.Length != Length) {
            throw new ArgumentException(
                $"Data contains {data.Length} elements, but the specified shape requires {Length}.",
                nameof(data)
            );
        }

        Buffer.CopyFromCPU(data);
    }

    /// <summary>
    /// Fills the tensor with a single value.
    /// </summary>
    public void Fill(float value) {
        float[] data = new float[Length];
        Array.Fill(data, value);
        Buffer.CopyFromCPU(data);
    }

    /// <summary>
    /// Creates a tensor filled with a given value.
    /// </summary>
    public static GpuTensor SingleValue(Accelerator accelerator, float value, params int[] shape) {
        GpuTensor tensor = new GpuTensor(accelerator, shape);
        tensor.Fill(value);
        return tensor;
    }

    /// <summary>
    /// Creates a tensor filled with zeros.
    /// </summary>
    public static GpuTensor Zero(Accelerator accelerator, params int[] shape) {
        return SingleValue(accelerator, 0, shape);
    }

    /// <summary>
    /// Creates a tensor filled with ones.
    /// </summary>
    public static GpuTensor One(Accelerator accelerator, params int[] shape) {
        return SingleValue(accelerator, 1, shape);
    }

    /// <summary>
    /// Creates a randomized tensor.
    /// </summary>
    public static GpuTensor Random(Accelerator accelerator, Random random, float min, float max,
        params int[] shape) {
        GpuTensor tensor = new GpuTensor(accelerator, shape);

        float[] data = new float[tensor.Length];
        for (int i = 0; i < data.Length; i++) {
            data[i] = min + (max - min) * random.NextSingle();
        }

        tensor.Buffer.CopyFromCPU(data);

        return tensor;
    }

    /// <summary>
    /// Released GPU memory owned by this tensor.
    /// </summary>
    public override void Dispose() {
        if (_disposed) return;

        Buffer.Dispose();
        _disposed = true;
    }
}

// Single Value

public interface IValueFloat : IDisposable { }

public class CpuValue : IValueFloat {

    /**
     * <summary>
     * Direct access to the CPU memory of the value.
     * Meant to be used by the Backend code.
     * </summary>
     */
    public float Value { get; set; }

    public CpuValue() : this(0) {
    }

    /**
     * <summary>
     * Create a value from existing data.
     * </summary>
     */
    public CpuValue(float value) {
        Value = value;
    }

    public static implicit operator float(CpuValue value) {
        return value?.Value ?? 0;
    }

    public void Dispose() {}
}

public class GpuValue : IValueFloat {

    private bool _disposed;
    
    /**
     * <summary>
     * The owning GPU buffer.
     * Meant to be used for Backend code.
     * </summary>
     */
    public MemoryBuffer1D<float, Stride1D.Dense> Buffer { get; }

    /**
     * <summary>
     * A non-owning vew of the GPU memory.
     * </summary>
     */
    public ArrayView1D<float, Stride1D.Dense> View => Buffer.View;

    public GpuValue(Accelerator accelerator) {
        ArgumentNullException.ThrowIfNull(accelerator);

        Buffer = accelerator.Allocate1D<float>(1);
    }
    /**
     * <summary>
     * Create a GPU tensor and initializes it with CPU data.
     * </summary>
     */
    public GpuValue(Accelerator accelerator, float data) : this(accelerator) {
        Buffer.CopyFromCPU([data]);
    }

    /**
     * <summary>
     * Released GPU memory owned by this tensor.
     * </summary>
     */
    public void Dispose() {
        if (_disposed) return;

        Buffer.Dispose();
        _disposed = true;
    }
}

// Integer Value

public interface IValueInt : IDisposable { }

public class CpuInt : IValueInt {

    /**
     * <summary>
     * Direct access to the CPU memory of the value.
     *
     * Meant to be used by the Backend code.
     * </summary>
     */
    public int Value { get; set; }

    public CpuInt() : this(0) { }

    /**
     * <summary>
     * Create a value from existing data.
     * </summary>
     */
    public CpuInt(int value) {
        Value = value;
    }

    public static implicit operator int(CpuInt value) {
        return value?.Value ?? 0;
    }

    public void Dispose() {}
}

public class GpuInt : IValueInt {

    private bool _disposed;
    
    /**
     * <summary>
     * The owning GPU buffer.
     * Meant to be used for Backend code.
     * </summary>
     */
    public MemoryBuffer1D<int, Stride1D.Dense> Buffer { get; }

    /**
     * <summary>
     * A non-owning vew of the GPU memory.
     * </summary>
     */
    public ArrayView1D<int, Stride1D.Dense> View => Buffer.View;

    public GpuInt(Accelerator accelerator) {
        ArgumentNullException.ThrowIfNull(accelerator);

        Buffer = accelerator.Allocate1D<int>(1);
    }
    /**
     * <summary>
     * Create a GPU tensor and initializes it with CPU data.
     * </summary>
     */
    public GpuInt(Accelerator accelerator, int data) : this(accelerator) {
        Buffer.CopyFromCPU([data]);
    }

    /**
     * <summary>
     * Released GPU memory owned by this tensor.
     * </summary>
     */
    public void Dispose() {
        if (_disposed) return;

        Buffer.Dispose();
        _disposed = true;
    }
}