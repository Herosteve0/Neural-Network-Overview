using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;

namespace NeuralNetwork.Core.Backend.Kernels;

using FloatView = ArrayView1D<float, Stride1D.Dense>;

using IntViewSingle = ArrayView1D<int, Stride1D.Dense>;

public class LossKernels(Accelerator accelerator) {
    public readonly Action<Index1D, IntViewSingle, IntViewSingle, IntViewSingle> CheckAnswer =
        accelerator.LoadAutoGroupedStreamKernel<Index1D, IntViewSingle, IntViewSingle, IntViewSingle>(CheckAnswerKernel);
    
    public readonly Action<Index1D, FloatView, IntViewSingle, float, FloatView> MseCalculate =
        accelerator.LoadAutoGroupedStreamKernel<Index1D, FloatView, IntViewSingle, float, FloatView>(MseCalculateKernel);
    public readonly Action<Index1D, FloatView, FloatView, float, FloatView> MseBackward =
        accelerator.LoadAutoGroupedStreamKernel<Index1D, FloatView, FloatView, float, FloatView>(MseBackwardKernel);
    
    public readonly Action<Index1D, FloatView, IntViewSingle, FloatView> CrossEntropyCalculate =
        accelerator.LoadAutoGroupedStreamKernel<Index1D, FloatView, IntViewSingle, FloatView>(CrossEntropyCalculateKernel);
    public readonly Action<Index1D, FloatView, FloatView, FloatView> CrossEntropyBackward =
        accelerator.LoadAutoGroupedStreamKernel<Index1D, FloatView, FloatView, FloatView>(CrossEntropyBackwardKernel);

    private static void CheckAnswerKernel(
        Index1D i,
        IntViewSingle prediction,
        IntViewSingle label,
        IntViewSingle accuracyOut
    ) {
        if (prediction[0] == label[0]) {
            accuracyOut[0]++;
        }
    }
    
    private static void MseCalculateKernel(
        Index1D i,
        FloatView input,
        IntViewSingle label,
        float a,
        FloatView lossOut
    ) {
        float loss = 0f;
        int labelIndex = label[0];
        for (int j = 0; j < input.Length; j++) {
            float expected = j == labelIndex ? 1f : 0f;
            float difference = input[j] - expected;
            loss += difference * difference;
        }
        lossOut[0] = a * loss;
    }
    
    private static void MseBackwardKernel(
        Index1D i,
        FloatView input,
        FloatView expectedInput,
        float a,
        FloatView output
    ) {
        output[i] = 2 * a * (input[i] - expectedInput[i]);
    }

    private static void CrossEntropyCalculateKernel(
        Index1D i,
        FloatView input,
        IntViewSingle label,
        FloatView lossOut
    ) {
        const float epsilon = 1e-15f;
        lossOut[i] = -XMath.Log(input[label[i]] + epsilon);
    }
    
    private static void CrossEntropyBackwardKernel(
        Index1D i,
        FloatView input,
        FloatView expectedInput,
        FloatView output
    ) {
        output[i] = input[i] - expectedInput[i];
    }
}