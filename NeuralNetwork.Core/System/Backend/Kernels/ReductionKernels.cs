using ILGPU;
using ILGPU.Runtime;
using NeuralNetwork.Core.System.Data;

namespace NeuralNetwork.Core.System.Backend.Kernels;

using FloatView = ArrayView1D<float, Stride1D.Dense>;
using IntView = ArrayView1D<int, Stride1D.Dense>;

public class ReductionKernels(Accelerator accelerator) {
	public readonly Action<Index1D, int, FloatView, FloatView> MaxReduction =
		accelerator.LoadAutoGroupedStreamKernel<Index1D, int, FloatView, FloatView>(MaxReductionKernel);
	public readonly Action<Index1D, int, FloatView, FloatView> MinReduction =
		accelerator.LoadAutoGroupedStreamKernel<Index1D, int, FloatView, FloatView>(MinReductionKernel);
	public readonly Action<Index1D, int, FloatView, FloatView> SumReduction =
		accelerator.LoadAutoGroupedStreamKernel<Index1D, int, FloatView, FloatView>(SumReductionKernel);

	public readonly Action<Index1D, int, FloatView, IntView, FloatView, IntView> MaxIndexReduction =
		accelerator.LoadAutoGroupedStreamKernel<Index1D, int, FloatView, IntView, FloatView, IntView>(
			MaxIndexReductionKernel);
		
	private static void SwapView<T>(ref T a, ref T b) {
		(a, b) = (b, a);
	}
	private void ReductionToValue(GpuTensor input, GpuValue output,
		Action<Index1D, int, FloatView, FloatView> kernel
	) {
		int count = input.Length;
			
		MemoryBuffer1D<float, Stride1D.Dense> inputBuffer = accelerator.Allocate1D<float>(count);
		MemoryBuffer1D<float, Stride1D.Dense> outputBuffer = accelerator.Allocate1D<float>((count + 1) / 2);
			
		input.View.CopyTo(inputBuffer.View);
			
		while (count > 1) {
			int outputSize = (count + 1) / 2;

			kernel(
				outputSize,
				count,
				inputBuffer.View,
				outputBuffer.View
			);

			SwapView(ref inputBuffer, ref outputBuffer);
				
			count = outputSize;
		}

		inputBuffer.View.SubView(0, 1)
			.CopyTo(output.View.SubView(0,1));
			
		accelerator.Synchronize();
			
		inputBuffer.Dispose();
		outputBuffer.Dispose();
	}
	private void ReductionToValueAndIndex(GpuTensor input, GpuValue outputValue, GpuInt outputIndex,
		Action<Index1D, int, FloatView, IntView, FloatView, IntView> kernel
	) {
		int count = input.Length;
			
		MemoryBuffer1D<float, Stride1D.Dense> inputValuesBuffer = accelerator.Allocate1D<float>(count);
		MemoryBuffer1D<int, Stride1D.Dense> inputIndicesBuffer = accelerator.Allocate1D<int>(count);
		MemoryBuffer1D<float, Stride1D.Dense> outputValuesBuffer = accelerator.Allocate1D<float>((count + 1) / 2);
		MemoryBuffer1D<int, Stride1D.Dense> outputIndicesBuffer = accelerator.Allocate1D<int>((count + 1) / 2);
			
		input.View.CopyTo(inputValuesBuffer.View);

		int[] indices = new int[count];
		for (int i = 0; i < count; i++) {
			indices[i] = i;
		}
		inputIndicesBuffer.View.CopyFromCPU(indices);
			
		while (count > 1) {
			int outputSize = (count + 1) / 2;

			kernel(
				outputSize,
				count,
				inputValuesBuffer.View,
				inputIndicesBuffer.View,
				outputValuesBuffer.View,
				outputIndicesBuffer.View
			);

			SwapView(ref inputValuesBuffer, ref outputValuesBuffer);
			SwapView(ref inputIndicesBuffer, ref outputIndicesBuffer);
				
			count = outputSize;
		}

		inputValuesBuffer.View.SubView(0, 1)
			.CopyTo(outputValue.View.SubView(0,1));
		inputIndicesBuffer.View.SubView(0, 1)
			.CopyTo(outputIndex.View.SubView(0,1));
			
		accelerator.Synchronize();
			
		inputValuesBuffer.Dispose();
		inputIndicesBuffer.Dispose();
		outputValuesBuffer.Dispose();
		outputIndicesBuffer.Dispose();
	}

	public void MaxIndex(GpuTensor input, GpuValue outputValue, GpuInt outputIndex) {
		ReductionToValueAndIndex(input, outputValue, outputIndex, MaxIndexReduction);
	}
	public void Max(GpuTensor input, GpuValue output) {
		ReductionToValue(input, output, MaxReduction);
	}
	public void Min(GpuTensor input, GpuValue output) {
		ReductionToValue(input, output, MinReduction);
	}

	public void Sum(GpuTensor input, GpuValue output) {
		ReductionToValue(input, output, SumReduction);
	}
	
	private static void MaxIndexReductionKernel(Index1D i, int length, FloatView inputValues, IntView inputIndices, FloatView outputValues, IntView outputIndices) {
		int left = i * 2;
		int right = left + 1;
			
		if (right >= length) {
			outputValues[i] = inputValues[left];
			outputIndices[i] = inputIndices[left];
			return;
		}

		if (inputValues[left] > inputValues[right]) {
			outputValues[i] = inputValues[left];
			outputIndices[i] = inputIndices[left];
		} else {
			outputValues[i] = inputValues[right];
			outputIndices[i] = inputIndices[right];
		}
	}
	private static void MaxReductionKernel(Index1D i, int length, FloatView input, FloatView output) {
		int left = i * 2;
		int right = left + 1;
			
		if (right >= length) {
			output[i] = input[left];
			return;
		}
			
		output[i] = MathF.Max(input[left], input[right]);
	}
	private static void MinReductionKernel(Index1D i, int length, FloatView input, FloatView output) {
		int left = i * 2;
		int right = left + 1;
			
		if (right >= length) {
			output[i] = input[left];
			return;
		}
			
		output[i] = MathF.Min(input[left], input[right]);
	}
	private static void SumReductionKernel(Index1D i, int length, FloatView input, FloatView output) {
		int left = i * 2;
		int right = left + 1;
			
		if (right >= length) {
			output[i] = input[left];
			return;
		}
			
		output[i] = input[left] + input[right];
	}
}