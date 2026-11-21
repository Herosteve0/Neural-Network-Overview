using System.Numerics;
using ILGPU;
using ILGPU.Runtime;

namespace NeuralNetwork.Core.Backend.Kernels;

using IntView = ArrayView1D<int, Stride1D.Dense>;
using FloatView = ArrayView1D<float, Stride1D.Dense>;

public class ElementwiseKernels(Accelerator accelerator) {
    public readonly Action<Index1D, FloatView, FloatView> Input =
        accelerator.LoadAutoGroupedStreamKernel<Index1D, FloatView, FloatView>(InputKernel);
    public readonly Action<Index1D, FloatView, FloatView, float> AdjustParameter =
        accelerator.LoadAutoGroupedStreamKernel<Index1D, FloatView, FloatView, float>(AdjustParameterKernel);

    public readonly Action<Index1D, IntView, IntView> IncreaseInt =
        accelerator.LoadAutoGroupedStreamKernel<Index1D, IntView, IntView>(IncreaseKernel);
    public readonly Action<Index1D, FloatView, FloatView> IncreaseFloat =
        accelerator.LoadAutoGroupedStreamKernel<Index1D, FloatView, FloatView>(IncreaseKernel);
    
    public readonly Action<Index1D, FloatView, FloatView, FloatView> Dropout =
        accelerator.LoadAutoGroupedStreamKernel<Index1D, FloatView, FloatView, FloatView>(DropoutKernel);

    private static void InputKernel(Index1D i, FloatView input, FloatView output) {
        const float mean = 0.1307f;
        const float std = 0.3081f;
        
        output[i] = (input[i] - mean) / std;
    }
    
    private static void AdjustParameterKernel(Index1D i, FloatView parameter, FloatView offset, float scalar) {
        parameter[i] -= offset[i] * scalar;
    }

    private static void IncreaseKernel<T>(
        Index1D i,
        ArrayView1D<T, Stride1D.Dense> input,
        ArrayView1D<T, Stride1D.Dense> offset
    ) where T : unmanaged, INumber<T> {
        input[i] += offset[i];
    }

    private static void DropoutKernel(Index1D i, FloatView input, FloatView mask, FloatView output) {
        output[i] = input[i] * mask[i];
    }
}