using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;

namespace NeuralNetwork.Core.Backend.Kernels;

using FloatView = ArrayView1D<float, Stride1D.Dense>;
using FloatViewSingle = ArrayView1D<float, Stride1D.Dense>;

using IntView = ArrayView1D<int, Stride1D.Dense>;
using IntViewSingle = ArrayView1D<int, Stride1D.Dense>;

public class ArithmeticKernels(Accelerator accelerator) {
	public readonly Action<Index1D, IntView, IntView, IntView> AdditionInt =
		accelerator.LoadAutoGroupedStreamKernel<Index1D, IntView, IntView, IntView>(AdditionKernel);
	
	public readonly Action<Index1D, FloatView, FloatView, FloatView> Addition =
		accelerator.LoadAutoGroupedStreamKernel<Index1D, FloatView, FloatView, FloatView>(AdditionKernel);

	public readonly Action<Index1D, FloatView, float, FloatView> ConstAddition =
		accelerator.LoadAutoGroupedStreamKernel<Index1D, FloatView, float, FloatView>(AdditionKernel);

	public readonly Action<Index1D, FloatView, FloatView, float, FloatView> ScalarAddition =
		accelerator.LoadAutoGroupedStreamKernel<Index1D, FloatView, FloatView, float, FloatView>(AdditionKernel);

	public readonly Action<Index1D, FloatView, FloatViewSingle, FloatView> GpuConstAddition =
		accelerator.LoadAutoGroupedStreamKernel<Index1D, FloatView, FloatViewSingle, FloatView>(
			GpuConstantAdditionKernel);

	public readonly Action<Index1D, IntViewSingle, IntViewSingle, IntViewSingle> IncreaseIfEqual =
		accelerator.LoadAutoGroupedStreamKernel<Index1D, IntViewSingle, IntViewSingle, IntViewSingle>(
			IncreaseIfEqualKernel);

	public readonly Action<Index1D, FloatView, FloatView, FloatView> Subtraction =
		accelerator.LoadAutoGroupedStreamKernel<Index1D, FloatView, FloatView, FloatView>(SubtractionKernel);
		
	public readonly Action<Index1D, FloatView, float,
		FloatView> ConstSubtraction = accelerator
		.LoadAutoGroupedStreamKernel<Index1D, FloatView, float, FloatView>(SubtractionKernel);
		
	public readonly Action<Index1D, FloatView, FloatView, float,
		FloatView> ScalarSubtraction = accelerator
		.LoadAutoGroupedStreamKernel<Index1D, FloatView, FloatView, float, FloatView>(SubtractionKernel);

	public readonly Action<Index1D, FloatView, FloatViewSingle,
		FloatView> GpuConstSubtraction = accelerator
		.LoadAutoGroupedStreamKernel<Index1D, FloatView, FloatViewSingle, FloatView>(GpuConstantSubtractionKernel);
		
		
	public readonly Action<Index1D, FloatView, FloatView,
		FloatView> ElementMultiplication = accelerator
		.LoadAutoGroupedStreamKernel<Index1D, FloatView, FloatView, FloatView>(ElementMultiplicationKernel);

	public readonly Action<Index1D, FloatView, float,
		FloatView> ConstMultiplication = accelerator
		.LoadAutoGroupedStreamKernel<Index1D, FloatView, float, FloatView>(ElementMultiplicationKernel);
		
	public readonly Action<Index1D, FloatView, FloatView,
		FloatView> GpuConstMultiplication = accelerator
		.LoadAutoGroupedStreamKernel<Index1D, FloatView, FloatViewSingle, FloatView>(GpuConstantElementMultiplicationKernel);
		
	public readonly Action<Index1D, FloatView, FloatView,
		FloatView> ElementDivision = accelerator
		.LoadAutoGroupedStreamKernel<Index1D, FloatView, FloatView, FloatView>(ElementDivisionKernel);

	public readonly Action<Index1D, FloatView, FloatView,
		FloatView> GpuConstDivision = accelerator
		.LoadAutoGroupedStreamKernel<Index1D, FloatView, FloatViewSingle, FloatView>(GpuConstantElementDivisionKernel);
		
	public readonly Action<Index1D, FloatView, float,
		FloatView> ConstDivision = accelerator
		.LoadAutoGroupedStreamKernel<Index1D, FloatView, float, FloatView>(ElementDivisionKernel);

		
	public readonly Action<Index1D, FloatView, FloatView> Exponentiate = accelerator
		.LoadAutoGroupedStreamKernel<Index1D, FloatView, FloatView>(ExponentiateKernel);

	public readonly Action<Index1D, FloatView, FloatViewSingle, FloatView> GpuConstantExponentiateSubtract =
		accelerator
			.LoadAutoGroupedStreamKernel<Index1D, FloatView, FloatViewSingle, FloatView>(
				GpuConstantExponentiateSubtractionKernel);
		
		
	public readonly Action<Index1D, FloatView, FloatView> Log = accelerator
		.LoadAutoGroupedStreamKernel<Index1D, FloatView, FloatView>(LogKernel);
	
	private static void AdditionKernel(
		Index1D i,
		IntView inputA,
		IntView inputB,
		IntView output
	) {
		output[i] = inputA[i] + inputB[i];
	}
	private static void AdditionKernel(
		Index1D i,
		FloatView inputA,
		FloatView inputB,
		FloatView output
	) {
		output[i] = inputA[i] + inputB[i];
	}
	private static void AdditionKernel(
		Index1D i,
		FloatView inputA,
		float constant,
		FloatView output
	) {
		output[i] = inputA[i] + constant;
	}
	private static void AdditionKernel(
		Index1D i,
		FloatView inputA,
		FloatView inputB,
		float constant,
		FloatView output
	) {
		output[i] = inputA[i] + inputB[i] * constant;
	}
	private static void GpuConstantAdditionKernel(
		Index1D i,
		FloatView inputA,
		FloatViewSingle constant,
		FloatView output
	) {
		output[i] = inputA[i] + constant[0];
	}

	private static void IncreaseIfEqualKernel(
		Index1D i,
		IntViewSingle inputA,
		IntViewSingle inputB,
		IntViewSingle output
	) {
		if (Math.Abs(inputA[0] - inputB[0]) < 0.1) {
			output[0]++;
		}
	}
		
	private static void SubtractionKernel(
		Index1D i,
		FloatView inputA,
		FloatView inputB,
		FloatView output
	) {
		output[i] = inputA[i] - inputB[i];
	}
	private static void SubtractionKernel(
		Index1D i,
		FloatView inputA,
		float constant,
		FloatView output
	) {
		output[i] = inputA[i] - constant;
	}
	private static void SubtractionKernel(
		Index1D i,
		FloatView inputA,
		FloatView inputB,
		float constant,
		FloatView output
	) {
		output[i] = inputA[i] - inputB[i] * constant;
	}
	private static void GpuConstantSubtractionKernel(
		Index1D i,
		FloatView inputA,
		FloatViewSingle constant,
		FloatView output
	) {
		output[i] = inputA[i] - constant[0];
	}

	private static void ElementMultiplicationKernel(
		Index1D i,
		FloatView inputA,
		FloatView inputB,
		FloatView output
	) {
		output[i] = inputA[i] * inputB[i];
	}
	private static void ElementMultiplicationKernel(
		Index1D i,
		FloatView inputA,
		float constant,
		FloatView output
	) {
		output[i] = inputA[i] * constant;
	}
	private static void GpuConstantElementMultiplicationKernel(
		Index1D i,
		FloatView inputA,
		FloatViewSingle constant,
		FloatView output
	) {
		output[i] = inputA[i] * constant[0];
	}

	private static void ElementDivisionKernel(
		Index1D i,
		FloatView inputA,
		FloatView inputB,
		FloatView output
	) {
		output[i] = inputA[i] / inputB[i];
	}
	private static void ElementDivisionKernel(
		Index1D i,
		FloatView inputA,
		float constant,
		FloatView output
	) {
		output[i] = inputA[i] / constant;
	}
	private static void GpuConstantElementDivisionKernel(
		Index1D i,
		FloatView inputA,
		FloatViewSingle constant,
		FloatView output
	) {
		output[i] = inputA[i] / constant[0];
	}

	private static void ExponentiateKernel(
		Index1D i,
		FloatView inputA,
		FloatView output
	) {
		output[i] = XMath.Exp(inputA[i]);
	}

	private static void GpuConstantExponentiateSubtractionKernel(
		Index1D i,
		FloatView inputA,
		FloatViewSingle constant,
		FloatView output
	) {
		output[i] = XMath.Exp(inputA[i] - constant[0]);
	}


	private static void LogKernel(
		Index1D i,
		FloatView input,
		FloatView output
	) {
		output[i] = XMath.Log(input[i]);
	}
}