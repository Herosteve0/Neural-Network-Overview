using ILGPU;
using ILGPU.Runtime;

namespace NeuralNetwork.Core.Backend.Kernels;

using FloatView = ArrayView1D<float, Stride1D.Dense>;

public class MatrixKernels(Accelerator accelerator) {
	public readonly Action<Index1D, int, FloatView, FloatView, FloatView, FloatView> Linear = accelerator
		.LoadAutoGroupedStreamKernel<Index1D, int, FloatView, FloatView, FloatView, FloatView>(LinearKernel);

	public readonly Action<Index1D, FloatView, FloatView, FloatView, int, int> Delta = accelerator
		.LoadAutoGroupedStreamKernel<Index1D, FloatView, FloatView, FloatView, int, int>(DeltaKernel);

	public readonly Action<Index1D, FloatView, FloatView, FloatView, int> WeightGradient = accelerator
		.LoadAutoGroupedStreamKernel<Index1D, FloatView, FloatView, FloatView, int>(WeightGradientKernel);
	public readonly Action<Index1D, FloatView, FloatView> BiasGradient = accelerator
		.LoadAutoGroupedStreamKernel<Index1D, FloatView, FloatView>(BiasGradientKernel);
	
	private static void LinearKernel(
		Index1D row, int columns, FloatView inputMatrix, FloatView inputVectorA, FloatView inputVectorB, FloatView output
	) {
		float sum = inputVectorB[row];

		int offset = row * columns;

		for (int col = 0; col < columns; col++) {
			sum += inputMatrix[offset + col] * inputVectorA[col];
		}

		output[row] = sum;
	}
	
	private static void DeltaKernel(
		Index1D col, FloatView delta, FloatView weights, FloatView deltaOut, int inputSize, int outputSize
	) {
		float sum = 0f;
		for (int row = 0; row < outputSize; row++) {
			sum += weights[row * inputSize + col] * delta[row];
		}
		deltaOut[col] = sum;
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