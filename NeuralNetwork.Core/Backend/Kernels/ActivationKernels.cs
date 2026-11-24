using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;

namespace NeuralNetwork.Core.Backend.Kernels;

using FloatView = ArrayView1D<float, Stride1D.Dense>;
using FloatViewSingle = ArrayView1D<float, Stride1D.Dense>;

public class ActivationKernels(Accelerator accelerator) {
    public readonly Action<Index1D, FloatView, FloatView> ReLuForward =
        accelerator.LoadAutoGroupedStreamKernel<Index1D, FloatView, FloatView>(ReLuForwardKernel);
    public readonly Action<Index1D, FloatView, FloatView> ReLuBackward =
        accelerator.LoadAutoGroupedStreamKernel<Index1D, FloatView, FloatView>(ReLuBackwardKernel);
    
    public readonly Action<Index1D, FloatView, float, FloatView> LeakyReLuForward =
        accelerator.LoadAutoGroupedStreamKernel<Index1D, FloatView, float, FloatView>(LeakyReLuForwardKernel);
    public readonly Action<Index1D, FloatView, float, FloatView> LeakyReLuBackward =
        accelerator.LoadAutoGroupedStreamKernel<Index1D, FloatView, float, FloatView>(LeakyReLuBackwardKernel);
    
    public readonly Action<Index1D, FloatView, FloatView> SigmoidForward =
        accelerator.LoadAutoGroupedStreamKernel<Index1D, FloatView, FloatView>(SigmoidForwardKernel);
    public readonly Action<Index1D, FloatView, FloatView> SigmoidBackward =
        accelerator.LoadAutoGroupedStreamKernel<Index1D, FloatView, FloatView>(SigmoidBackwardKernel);
    
    public readonly Action<Index1D, FloatView, FloatViewSingle, FloatView> SoftmaxExp =
        accelerator.LoadAutoGroupedStreamKernel<Index1D, FloatView, FloatViewSingle, FloatView>(SoftmaxExpKernel);
    public readonly Action<Index1D, FloatView, FloatViewSingle> SoftmaxDivide =
        accelerator.LoadAutoGroupedStreamKernel<Index1D, FloatView, FloatViewSingle>(SoftmaxDivideKernel);
    public readonly Action<Index1D, int, FloatView, FloatView> SoftMaxBackward =
        accelerator.LoadAutoGroupedStreamKernel<Index1D, int, FloatView, FloatView>(SoftmaxBackwardKernel);
    
    private static void ReLuForwardKernel(Index1D i, FloatView input, FloatView output) {
        output[i] = Math.Max(input[i], 0);
    }

    private static void ReLuBackwardKernel(Index1D i, FloatView input, FloatView output) {
        output[i] *= input[i] > 0 ? 1 : 0;
    }

    private static void LeakyReLuForwardKernel(Index1D i, FloatView input, float a, FloatView output) {
        output[i] = input[i] * (input[i] > 0 ? 1 : a);
    }
    
    private static void LeakyReLuBackwardKernel(Index1D i, FloatView input, float a, FloatView output) {
        output[i] *= input[i] > 0 ? 1 : a;
    }

    private static void SigmoidForwardKernel(Index1D i, FloatView input, FloatView output) {
        float e = XMath.Exp(-input[i]);
        output[i] = 1 / (1 + e);
    }
    
    private static void SigmoidBackwardKernel(Index1D i, FloatView input, FloatView output) {
        float e = XMath.Exp(-input[i]);
        output[i] *= e / ((1 + e) * (1 + e));
    }

    private static void SoftmaxExpKernel(Index1D i, FloatView input, FloatViewSingle max, FloatView output) {
        output[i] = XMath.Exp(input[i] - max[0]);
    }

    private static void SoftmaxDivideKernel(Index1D i, FloatView output, FloatViewSingle sum) {
        output[i] /= sum[0];
    }

    private static void SoftmaxBackwardKernel(Index1D i, int length, FloatView input, FloatView output) {
        // Doesn't work, don't want to make it work :)
        float max = input[0];
        for (int j = 1; j < length; j++) max = Math.Max(max, input[j]);

        float sum = 0f;
        for (int j = 0; j < length; j++) sum += XMath.Exp(input[j] - max);

        float dot = 0f;
        for (int j = 0; j < length; j++) {
            float probability = XMath.Exp(input[j] - max) / sum;
            dot += output[j] * probability;
        }

        float probabilityAtIndex = XMath.Exp(input[i] - max) / sum;
        output[i] = probabilityAtIndex * (output[i] - dot);
    }
}