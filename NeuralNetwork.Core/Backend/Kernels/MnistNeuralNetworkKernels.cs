using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;

namespace NeuralNetwork.Core.Backend.Kernels;

using FloatView = ArrayView1D<float, Stride1D.Dense>;
using FloatViewSingle = ArrayView1D<float, Stride1D.Dense>;
using IntViewSingle = ArrayView1D<int, Stride1D.Dense>;

public class MnistNeuralNetworkKernels(Accelerator accelerator) {
	public readonly Action<Index1D, FloatView, IntViewSingle, FloatViewSingle> Loss = accelerator
		.LoadAutoGroupedStreamKernel<Index1D, FloatView, IntViewSingle, FloatViewSingle>(LossKernel);
	
	public readonly Action<Index1D, FloatView, FloatView> Input = accelerator
		.LoadAutoGroupedStreamKernel<Index1D, FloatView, FloatView>(InputKernel);

	public readonly Action<Index1D, FloatView, FloatView, FloatView, FloatView, int> Backward = accelerator
		.LoadAutoGroupedStreamKernel<Index1D, FloatView, FloatView, FloatView, FloatView, int>(BackwardKernel);

	public readonly Action<Index1D, FloatView, FloatView, FloatView, int> WeightGradient = accelerator
		.LoadAutoGroupedStreamKernel<Index1D, FloatView, FloatView, FloatView, int>(WeightGradientKernel);
	public readonly Action<Index1D, FloatView, FloatView> BiasGradient = accelerator
		.LoadAutoGroupedStreamKernel<Index1D, FloatView, FloatView>(BiasGradientKernel);

	private static void LossKernel(Index1D i, FloatView activations, IntViewSingle label, FloatViewSingle lossOut) {
		const float epsilon = 1e-15f;
		
		lossOut[i] = -XMath.Log(activations[label[i]] + epsilon);
	}
	
	private static void InputKernel(Index1D i, FloatView input, FloatView activationOut) {
		const float mean = 0.1307f;
		const float std = 0.3081f;
			
		activationOut[i] = (input[i] - mean) / std;
	}
	
	private static void BackwardKernel(
		Index1D row, FloatView values, FloatView delta, FloatView weightsT, FloatView deltaOut, int deltaSize
	) {
		if (values[row] <= 0f) {
			deltaOut[row] = 0f;
			return;
		}
		
		int offset = row * deltaSize;

		float sum = 0;
		for (int col = 0; col < deltaSize; col++) {
			sum += weightsT[offset + col] * delta[col];
		}
			
		deltaOut[row] = sum;
	}
		
	private static void WeightGradientKernel(
		Index1D i, FloatView delta, FloatView activations, FloatView weightDelta, int columns
	) {
		int row = i / columns;
		int col = i % columns;
			
		weightDelta[i] += delta[row] * activations[col];
	}
	private static void BiasGradientKernel(Index1D i, FloatView delta, FloatView biasDelta) {
		biasDelta[i] += delta[i];
	}
}