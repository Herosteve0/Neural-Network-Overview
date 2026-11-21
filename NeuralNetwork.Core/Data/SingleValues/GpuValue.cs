using ILGPU;
using ILGPU.Runtime;

namespace NeuralNetwork.Core.Data.SingleValues;

public class GpuValue<T> : ISingleValue<T> where T : unmanaged{
    private bool _disposed;
    
    /**
     * <summary>
     * The owning GPU buffer.
     * Meant to be used for Backend code.
     * </summary>
     */
    public MemoryBuffer1D<T, Stride1D.Dense> Buffer { get; }

    /**
     * <summary>
     * A non-owning vew of the GPU memory.
     * </summary>
     */
    public ArrayView1D<T, Stride1D.Dense> View => Buffer.View;

    public GpuValue(Accelerator accelerator) {
        ArgumentNullException.ThrowIfNull(accelerator);

        Buffer = accelerator.Allocate1D<T>(1);
    }

    /**
     * <summary>
     * Create a GPU tensor and initializes it with CPU data.
     * </summary>
     */
    public GpuValue(Accelerator accelerator, T data) : this(accelerator) {
        try {
            Buffer.CopyFromCPU([data]);
        } catch {
            Buffer.Dispose();
            _disposed = true;
            throw;
        }
    }

    /**
     * <summary>
     * Released GPU memory owned by this tensor.
     * </summary>
     */
    public void Dispose() {
        if (_disposed) return;
        GC.SuppressFinalize(this);

        Buffer.Dispose();
        _disposed = true;
    }
}