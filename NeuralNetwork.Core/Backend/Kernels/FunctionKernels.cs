using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;

namespace NeuralNetwork.Core.Backend.Kernels;

using FloatView = ArrayView1D<float, Stride1D.Dense>;

public class FunctionKernels(Accelerator accelerator) {
	public readonly Action<Index1D, FloatView, FloatView> ReLu = accelerator
		.LoadAutoGroupedStreamKernel<Index1D, FloatView, FloatView>(ReLuKernel);
	public readonly Action<Index1D, FloatView, FloatView> LeakyReLu = accelerator
		.LoadAutoGroupedStreamKernel<Index1D, FloatView, FloatView>(LeakyReLuKernel);
	public readonly Action<Index1D, FloatView, FloatView> Sigmoid = accelerator
		.LoadAutoGroupedStreamKernel<Index1D, FloatView, FloatView>(SigmoidKernel);
		
	private static void ReLuKernel(Index1D i, FloatView input, FloatView output) {
		output[i] = MathF.Max(input[i], 0);
	}
	private static void ReLuDerivativeKernel(Index1D i, FloatView input, FloatView output) {
		output[i] = input[i] > 0 ? 1 : 0;
	}

	private const float LeakyReLuConstant = 0.15f;
	private static void LeakyReLuKernel(Index1D i, FloatView input, FloatView output) {
		output[i] = input[i] > 0 ? input[i] : LeakyReLuConstant * input[i];
	}
	private static void LeakyReLuDerivativeKernel(Index1D i, FloatView input, FloatView output) {
		output[i] = input[i] > 0 ? 1 : LeakyReLuConstant;
	}
		
	private static void SigmoidKernel(Index1D i, FloatView input, FloatView output) {
		output[i] = 1 / (XMath.Exp(-input[i]) + 1);
	}
	private static void SigmoidDerivativeKernel(Index1D i, FloatView input, FloatView output) {
		float e = XMath.Exp(input[i]);
		output[i] = e / ((e + 1) * (e + 1));
	}
}