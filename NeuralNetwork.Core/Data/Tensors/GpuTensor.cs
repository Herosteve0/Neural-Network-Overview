using ILGPU;
using ILGPU.Runtime;

namespace NeuralNetwork.Core.Data.Tensors;

public class GpuTensor<T> : BaseTensor<T> where T : unmanaged {

    private bool _disposed;

    /// <summary>
    /// The owning GPU buffer.
    ///
    /// Meant to be used for Backend code.
    /// </summary>
    public MemoryBuffer1D<T, Stride1D.Dense> Buffer { get; }

    /// <summary>
    /// A non-owning vew of the GPU memory.
    /// </summary>
    public ArrayView1D<T, Stride1D.Dense> View => Buffer.View;

    /// <summary>
    /// Create a GPU tensor.
    /// </summary>
    public GpuTensor(Accelerator accelerator, params int[] shape) : base(shape) {
        ArgumentNullException.ThrowIfNull(accelerator);

        Buffer = accelerator.Allocate1D<T>(Length);
    }

    /// <summary>
    /// Create a GPU tensor and initializes it with CPU data.
    /// </summary>
    public GpuTensor(Accelerator accelerator, T[] data, params int[] shape) : this(accelerator, shape) {
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
    /// Creates a GPU tensor filled with a given value.
    /// </summary>
    public GpuTensor(Accelerator accelerator, T value, params int[] shape) : this(accelerator, shape) {
        T[] data = new T[Length];
        Array.Fill(data, value);
        Buffer.CopyFromCPU(data);
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