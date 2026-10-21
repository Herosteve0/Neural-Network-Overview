using ILGPU;
using ILGPU.Runtime;

namespace NeuralNetwork.Core.System.Backend.Kernels;

using FloatView = ArrayView1D<float, Stride1D.Dense>;

public class MatrixKernels(Accelerator accelerator) {
	/**
	 * <param name="Rows">
	 * Rows of the matrix
	 * </param>
	 * <param name="Columns">
	 * Columns of the matrix
	 * </param>
	 * <param name="Matrix">
	 * A Matrix "A", size Rows x Columns
	 * </param>
	 * <param name="Scaled Vector">
	 * A Vector "x", size Columns
	 * </param>
	 * <param name="Addition Vector">
	 * A Vector "b", size Rows
	 * </param>
	 * <param name="Output Vector">
	 * A Vector "out", size Rows
	 * </param>
	 *
	 * <returns>
	 * out = A * x + b
	 * </returns>
	 */
	public readonly Action<Index1D, int, FloatView, FloatView, FloatView, FloatView> Linear = accelerator
		.LoadAutoGroupedStreamKernel<Index1D, int, FloatView, FloatView, FloatView, FloatView>(LinearKernel);

	public readonly Action<Index1D, int, int, FloatView, FloatView> Transpose = accelerator
		.LoadAutoGroupedStreamKernel<Index1D, int, int, FloatView, FloatView>(TransposeKernel);
		
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

	private static void TransposeKernel(Index1D i, int rows, int columns, FloatView input, FloatView output) {
		int row = i / columns;
		int col = i % columns;
			
		output[col * rows + row] = input[row * columns + col];
	}
}