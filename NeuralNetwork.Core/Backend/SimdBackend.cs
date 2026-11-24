using System.Numerics;
using NeuralNetwork.Core.Data.SingleValues;
using NeuralNetwork.Core.Data.Tensors;

namespace NeuralNetwork.Core.Backend;

/**
 * Cpu Simd powered backend.
 */
public class SimdBackend : CpuBackend {
	private readonly int _simdWidth = Vector<float>.Count;

	public SimdBackend() {
		Console.WriteLine(
			$"Using Simd Backend with width {_simdWidth} floats. ({_simdWidth * sizeof(float) * 8} bits)");
	}

	protected override void CpuInput(CpuTensor<float> input, CpuTensor<float> activationOut) {
		const float mean = 0.1307f;
		const float std = 0.3081f;

		Vector<float> vMean = new(mean);
		Vector<float> vStd = new(std);

		ReadOnlySpan<float> inputSpan = input.Data;

		Span<float> activationOutSpan = activationOut.Data;

		int i = 0;
		for (; i <= input.Length - _simdWidth; i += _simdWidth) {
			Vector<float> vInput = new(inputSpan.Slice(i, _simdWidth));
			((vInput - vMean) / vStd).CopyTo(activationOutSpan.Slice(i, _simdWidth));
		}

		for (; i < input.Length; i++) {
			activationOutSpan[i] = (input[i] - mean) / std;
		}
	}

	protected override void CpuMaxIndex(CpuTensor<float> input, CpuValue<int> index) {
		index.Value = MaxVectorIndex(input).index;
	}
	
	protected override void CpuLinear(CpuTensor<float> input, CpuTensor<float> weights, CpuTensor<float> bias,
		CpuTensor<float> valuesOut) {
		int rows = weights.Shape[0];
		int columns = weights.Shape[1];

		ReadOnlySpan<float> weightsSpan = weights.Data;
		ReadOnlySpan<float> inputSpan = input.Data;
		ReadOnlySpan<float> biasSpan = bias.Data;

		Span<float> valuesOutSpan = valuesOut.Data;

		for (int row = 0; row < rows; row++) {
			int offset = row * columns;

			Vector<float> vSum = Vector<float>.Zero;

			int col = 0;
			for (; col <= columns - _simdWidth; col += _simdWidth) {
				Vector<float> vWeights = new(weightsSpan.Slice(offset + col, _simdWidth));
				Vector<float> vInput = new(inputSpan.Slice(col, _simdWidth));
				vSum += vWeights * vInput;
			}

			float sum = biasSpan[row] + Vector.Sum(vSum);

			for (; col < columns; col++) {
				sum += weightsSpan[offset + col] * inputSpan[col];
			}

			valuesOutSpan[row] = sum;
		}
	}

	protected override void CpuDelta(CpuTensor<float> delta, CpuTensor<float> weights, CpuTensor<float> deltaOut) {
		int rows = weights.Shape[0];
		int columns = weights.Shape[1];

		ReadOnlySpan<float> deltaSpan = delta.Data;
		ReadOnlySpan<float> weightsSpan = weights.Data;

		Span<float> deltaOutSpan = deltaOut.Data;

		Span<float> buffer = new float[_simdWidth];

		for (int col = 0; col < columns; col++) {
			Vector<float> vSum = Vector<float>.Zero;
			
			int row = 0;
			for (; row < rows - _simdWidth; row += _simdWidth) {
				for (int i = 0; i < _simdWidth; i++) {
					buffer[i] = weightsSpan[(row + i) * columns + col];
				}
				
				Vector<float> vWeights = new(buffer);
				Vector<float> vDelta = new(deltaSpan.Slice(row, _simdWidth));
				vSum += vWeights * vDelta;
			}

			float sum = Vector.Sum(vSum);
			
			for (; row < rows; row++) {
				sum += weightsSpan[row * columns + col] * deltaSpan[row];
			}
			
			deltaOutSpan[col] = sum;
		}
	}

	protected override void CpuBackPropagation(CpuTensor<float> delta, CpuTensor<float> activations,
		CpuTensor<float> weightDelta, CpuTensor<float> biasDelta) {
		int rows = delta.Shape[0];
		int columns = activations.Shape[0];

		ReadOnlySpan<float> deltaSpan = delta.Data;
		ReadOnlySpan<float> activationsSpan = activations.Data;

		Span<float> weightDeltaSpan = weightDelta.Data;
		Span<float> biasDeltaSpan = biasDelta.Data;

		int row = 0;
		for (; row < rows; row++) {
			int offset = row * columns;

			int col = 0;
			for (; col <= columns - _simdWidth; col += _simdWidth) {
				Vector<float> vActivation = new(activationsSpan.Slice(col, _simdWidth));
				Vector<float> vWeightsDelta = new(weightDeltaSpan.Slice(offset + col, _simdWidth));

				(vActivation * deltaSpan[row] + vWeightsDelta).CopyTo(weightDeltaSpan.Slice(offset + col, _simdWidth));
			}

			for (; col < columns; col++) {
				weightDeltaSpan[offset + col] += deltaSpan[row] * activationsSpan[col];
			}
		}

		row = 0;
		for (; row < rows - _simdWidth; row += _simdWidth) {
			Vector<float> vDelta = new(deltaSpan.Slice(row, _simdWidth));
			Vector<float> vBiasDelta = new(biasDeltaSpan.Slice(row, _simdWidth));

			(vBiasDelta + vDelta).CopyTo(biasDeltaSpan.Slice(row, _simdWidth));
		}

		for (; row < rows; row++) {
			biasDeltaSpan[row] += deltaSpan[row];
		}
	}

	protected override void CpuAdjust(CpuTensor<float> weightDelta, CpuTensor<float> biasDelta, float scale,
		CpuTensor<float> weights, CpuTensor<float> bias) {
		ReadOnlySpan<float> weightDeltaSpan = weightDelta.Data;
		ReadOnlySpan<float> biasDeltaSpan = biasDelta.Data;

		Span<float> weightsSpan = weights.Data;
		Span<float> biasSpan = bias.Data;

		int length = weights.Length;
		int i = 0;
		for (; i <= length - _simdWidth; i += _simdWidth) {
			Vector<float> vWeights = new(weightsSpan.Slice(i, _simdWidth));
			Vector<float> vWeightDelta = new(weightDeltaSpan.Slice(i, _simdWidth));
			(vWeights - vWeightDelta * scale).CopyTo(weightsSpan.Slice(i, _simdWidth));
		}

		for (; i < length; i++) {
			weightsSpan[i] -= weightDeltaSpan[i] * scale;
		}

		length = bias.Length;
		i = 0;
		for (; i <= length - _simdWidth; i += _simdWidth) {
			Vector<float> vBias = new(biasSpan.Slice(i, _simdWidth));
			Vector<float> vBiasDelta = new(biasDeltaSpan.Slice(i, _simdWidth));
			(vBias - vBiasDelta * scale).CopyTo(biasSpan.Slice(i, _simdWidth));
		}

		for (; i < length; i++) {
			biasSpan[i] -= biasDeltaSpan[i] * scale;
		}
	}

	protected override void CpuDropout(CpuTensor<float> input, CpuTensor<float> mask, CpuTensor<float> output) {
		int length = input.Length;
		
		ReadOnlySpan<float> inputSpan = input.Data;
		ReadOnlySpan<float> maskSpan = mask.Data;
		Span<float> outputSpan = output.Data;

		int i = 0;
		for (; i <= length - _simdWidth; i += _simdWidth) {
			Vector<float> vInput = new(inputSpan.Slice(i, _simdWidth));
			Vector<float> vMask = new(maskSpan.Slice(i, _simdWidth));
			
			(vInput * vMask).CopyTo(outputSpan.Slice(i, _simdWidth));
		}

		for (; i < length; i++) {
			outputSpan[i] = inputSpan[i] * maskSpan[i];
		}
	}

	protected override void CpuReLuForward(CpuTensor<float> input, CpuTensor<float> output) {
		int length = input.Length;

		ReadOnlySpan<float> inputSpan = input.Data;

		Span<float> outputSpan = output.Data;

		Vector<float> vZero = Vector<float>.Zero;

		int i = 0;
		for (; i <= length - _simdWidth; i += _simdWidth) {
			Vector<float> vInput = new(inputSpan.Slice(i, _simdWidth));
			Vector.Max(vInput, vZero).CopyTo(outputSpan.Slice(i, _simdWidth));
		}

		for (; i < length; i++) {
			outputSpan[i] = Math.Max(inputSpan[i], 0);
		}
	}

	protected override void CpuReLuBackward(CpuTensor<float> input, CpuTensor<float> gradient) {
		int length = input.Length;

		ReadOnlySpan<float> inputSpan = input.Data;

		Span<float> gradientSpan = gradient.Data;

		Vector<float> vZero = Vector<float>.Zero;

		int i = 0;
		for (; i <= length - _simdWidth; i += _simdWidth) {
			Vector<float> vInput = new(inputSpan.Slice(i, _simdWidth));
			Vector<float> vOutput = new(gradientSpan.Slice(i, _simdWidth));
			Vector<int> vMask = Vector.GreaterThan(vInput, vZero);

			Vector.ConditionalSelect(vMask, vOutput, vZero)
				.CopyTo(gradientSpan.Slice(i, _simdWidth));
		}

		for (; i < length; i++) {
			if (inputSpan[i] >= 0) continue;
			gradientSpan[i] = 0;
		}
	}

	protected override void CpuLeakyReLuForward(CpuTensor<float> input, float a, CpuTensor<float> output) {
		int length = input.Length;
		
		ReadOnlySpan<float> inputSpan = input.Data;
		
		Span<float> outputSpan = output.Data;

		Vector<float> vZero = Vector<float>.Zero;
		Vector<float> vOne = Vector<float>.One;
		Vector<float> vA = new(a);
		
		int i = 0;
		for (; i <= length - _simdWidth; i += _simdWidth) {
			Vector<float> vInput = new(inputSpan.Slice(i, _simdWidth));
			Vector<int> vMask = Vector.GreaterThan(vInput, vZero);

			Vector<float> vOffset = Vector.ConditionalSelect(vMask, vOne, vA);
			(vInput * vOffset).CopyTo(outputSpan.Slice(i, _simdWidth));
		}

		for (; i < length; i++) {
			outputSpan[i] = inputSpan[i] * (inputSpan[i] > 0 ? 1f : a);
		}
	}

	protected override void CpuLeakyReLuBackward(CpuTensor<float> input, float a, CpuTensor<float> gradient) {
		int length = input.Length;
		
		ReadOnlySpan<float> inputSpan = input.Data;
		
		Span<float> gradientSpan = gradient.Data;

		Vector<float> vZero = Vector<float>.Zero;
		Vector<float> vOne = Vector<float>.One;
		Vector<float> vA = new(a);
		
		int i = 0;
		for (; i <= length - _simdWidth; i += _simdWidth) {
			Vector<float> vInput = new(inputSpan.Slice(i, _simdWidth));
			Vector<float> vOutput = new(gradientSpan.Slice(i, _simdWidth));
			Vector<int> vMask = Vector.GreaterThan(vInput, vZero);

			Vector<float> vOffset = Vector.ConditionalSelect(vMask, vOne, vA);
			(vOutput * vOffset).CopyTo(gradientSpan.Slice(i, _simdWidth));
		}

		for (; i < length; i++) {
			gradientSpan[i] *= input[i] > 0 ? 1f : a;
		}
	}

	protected override void CpuSoftmaxForward(CpuTensor<float> input, CpuTensor<float> output,
		CpuValue<int> predictionOut) {
		ReadOnlySpan<float> inputSpan = input.Data;
			
		Span<float> outputSpan = output.Data;
			
		// Obtaining Max

		(int maxIndex, float max) = MaxVectorIndex(input);

		// Creating each e^(...) value and the sum of them all
		Vector<float> vMax =  new(max);
		Vector<float> vSum = Vector<float>.Zero;
			
		int i;
		for (i = 0; i <= input.Length - _simdWidth; i += _simdWidth) {
			Vector<float> vInput = new (inputSpan.Slice(i, _simdWidth));
			Vector<float> vE = Vector.Exp(vInput - vMax);
			
			vE.CopyTo(outputSpan.Slice(i, _simdWidth));
			vSum += vE;
		}

		float sum = Vector.Sum(vSum);
			
		for (; i < input.Length; i++) {
			float e = (float)Math.Exp(inputSpan[i] - max);
			outputSpan[i] = e;
			sum += e;
		}

		// Divide the final output with the sum
			
		for (i = 0; i < output.Length - _simdWidth; i += _simdWidth) {
			Vector<float> vOutput = new (outputSpan.Slice(i, _simdWidth));
			(vOutput / sum).CopyTo(outputSpan.Slice(i, _simdWidth));
		}

		for (; i < output.Length; i++) {
			outputSpan[i] /= sum;
		}

		predictionOut?.Value = maxIndex;
	}

	protected override void CpuSoftmaxBackward(CpuTensor<float> input, CpuTensor<float> gradient) {
		int length = input.Length;
		
		ReadOnlySpan<float> inputSpan = input.Data;
		Span<float> gradientSpan = gradient.Data;

		int i;
		
		// Max

		float max = MaxVectorIndex(input).value;
		Vector<float> vMax = new(max);
		
		// Sum: Σ e^(input[i] - max)
		
		Vector<float> vSum = Vector<float>.Zero;
		for (i = 0; i <= length - _simdWidth; i += _simdWidth) {
			Vector<float> vInput = new (inputSpan.Slice(i, _simdWidth));
			vSum += Vector.Exp(vInput - vMax);
		}
		
		float sum = Vector.Sum(vSum);
		for (; i < length; i++) {
			sum += (float)Math.Exp(inputSpan[i] - max);
		}

		vSum = new Vector<float>(sum);
		
		// Dot: Σ (gradient[i] * Softmax(i))

		Vector<float> vDot = Vector<float>.Zero;
		for (i = 0; i <= length - _simdWidth; i += _simdWidth) {
			Vector<float> vInput = new (inputSpan.Slice(i, _simdWidth));
			Vector<float> vGradient = new (gradientSpan.Slice(i, _simdWidth));
			
			Vector<float> vProbability = Vector.Exp(vInput - vMax) / vSum;
			
			vDot += vGradient * vProbability;
		}
		
		float dot = Vector.Sum(vDot);
		for (; i < length; i++) {
			float probability = (float)Math.Exp(inputSpan[i] - max) / sum;
			dot += gradientSpan[i] * probability;
		}
		vDot = new Vector<float>(dot);
		
		// Softmax(i) * (gradient[i] - dot)

		for (i = 0; i <= length - _simdWidth; i += _simdWidth) {
			Vector<float> vInput = new (inputSpan.Slice(i, _simdWidth));
			Vector<float> vGradient = new (gradientSpan.Slice(i, _simdWidth));
			
			Vector<float> vProbability = Vector.Exp(vInput - vMax) / vSum;

			(vProbability*(vGradient - vDot)).CopyTo(gradientSpan.Slice(i, _simdWidth));
		}

		for (; i < length; i++) {
			float probability = (float)Math.Exp(inputSpan[i] - max) / sum;
			gradient[i] = probability * (gradient[i] - dot);
		}
	}

	protected override void CpuSigmoidForward(CpuTensor<float> input, CpuTensor<float> output) {
		int length = input.Length;
		
		ReadOnlySpan<float> inputSpan = input.Data;
		Span<float> outputSpan = output.Data;

		Vector<float> vOne = Vector<float>.One;
		
		int i = 0;
		for (; i <= length - _simdWidth; i += _simdWidth) {
			Vector<float> vInput = new (inputSpan.Slice(i, _simdWidth));

			Vector<float> vE = Vector.Exp(-vInput);
			(vOne / (vOne + vE)).CopyTo(outputSpan.Slice(i, _simdWidth));
		}

		for (; i < length; i++) {
			float e = (float)Math.Exp(-input[i]);
			outputSpan[i] = 1 / (1 + e);
		}
	}

	protected override void CpuSigmoidBackward(CpuTensor<float> input, CpuTensor<float> gradient) {
		int length = input.Length;
		
		ReadOnlySpan<float> inputSpan = input.Data;
		Span<float> gradientSpan = gradient.Data;

		Vector<float> vOne = Vector<float>.One;
		
		int i = 0;
		for (; i <= length - _simdWidth; i += _simdWidth) {
			Vector<float> vInput = new (inputSpan.Slice(i, _simdWidth));
			Vector<float> vOutput = new (gradientSpan.Slice(i, _simdWidth));
			
			Vector<float> vE = Vector.Exp(-Vector.Abs(vInput));
			Vector<float> vOffset = vE / ((vOne + vE) * (vOne + vE));
			(vOutput * vOffset).CopyTo(gradientSpan.Slice(i, _simdWidth));
		}

		for (; i < length; i++) {
			float e = (float)Math.Exp(-Math.Abs(inputSpan[i]));
			gradientSpan[i] *= e / ((1 + e) * (1 + e));
		}
	}

	
	protected override void CpuMseCalculation(CpuTensor<float> activation, CpuValue<int> label, float a, CpuValue<float> lossOut) {
		int length = activation.Length;

		ReadOnlySpan<float> activationSpan = activation.Data;
		
		int[] laneIndices = new int[_simdWidth];
		for (int k = 0; k < _simdWidth; k++) laneIndices[k] = k;
		Vector<int> vIndex = new (laneIndices);
		Vector<int> vSteps = new (_simdWidth);
		
		Vector<int> vLabel = new(label);
		
		Vector<float> vZero = Vector<float>.Zero;
		Vector<float> vOne = Vector<float>.One;
		
		lossOut.Value = 0;

		int i = 0;
		for (; i <= length - _simdWidth; i += _simdWidth) {
			Vector<float> vActivation =  new(activationSpan.Slice(i, _simdWidth));
			Vector<int> vMask = Vector.Equals(vIndex, vLabel);
			Vector<float> vExpected = Vector.ConditionalSelect(vMask, vOne, vZero);
			
			Vector<float> vValue = vActivation - vExpected;
			
			lossOut.Value += Vector.Sum(vValue * vValue);
			
			vIndex += vSteps;
		}

		for (; i < length; i++) {
			float value = i == label ? 1 : 0;
			lossOut.Value += (activation[i] - value) * (activation[i] - value);
		}
		lossOut.Value *= a;
	}

	protected override void CpuMseBackward(CpuTensor<float> input, CpuTensor<float> expectedInput, float a, CpuTensor<float> output) {
		int length = input.Length;
		
		ReadOnlySpan<float> inputSpan = input.Data;
		ReadOnlySpan<float> expectedInputSpan = expectedInput.Data;
		Span<float> outputSpan = output.Data;

		Vector<float> vA = new(a);

		int i = 0;
		for (; i <= length - _simdWidth; i += _simdWidth) {
			Vector<float> vInput = new (inputSpan.Slice(i, _simdWidth));
			Vector<float> vExpectedInput = new (expectedInputSpan.Slice(i, _simdWidth));
			
			Vector<float> vOffset = vInput - vExpectedInput;
			(2 * vA * vOffset).CopyTo(outputSpan.Slice(i, _simdWidth));
		}

		for (; i < length; i++) {
			outputSpan[i] = 2 * a * (inputSpan[i] - expectedInputSpan[i]);
		}
	}

	protected override void CpuCrossEntropyCalculate(CpuTensor<float> activation, CpuValue<int> label, CpuValue<float> lossOut) {
		const float epsilon = 1e-15f;
		lossOut.Value = -(float)Math.Log(activation[label] + epsilon);
	}

	protected override void CpuCrossEntropyBackward(CpuTensor<float> input, CpuTensor<float> expectedInput, CpuTensor<float> output) {
		int length = input.Length;
			
		ReadOnlySpan<float> inputSpan = input.Data;
		ReadOnlySpan<float> expectedValuesSpan = expectedInput.Data;
		Span<float> outputSpan = output.Data;

		int i = 0;
		for (; i <= length - _simdWidth; i += _simdWidth) {
			Vector<float> vInput = new (inputSpan.Slice(i, _simdWidth));
			Vector<float> vExpectedValues = new (expectedValuesSpan.Slice(i, _simdWidth));
				
			(vInput - vExpectedValues).CopyTo(outputSpan.Slice(i, _simdWidth));
		}

		for (; i < length; i++) {
			outputSpan[i] = inputSpan[i] - expectedValuesSpan[i];
		}
	}


	private (int index, float value) MaxVectorIndex(CpuTensor<float> a) {
		ReadOnlySpan<float> span = a.Data;
			
		int[] laneIndices = new int[_simdWidth];
		for (int k = 0; k < _simdWidth; k++) laneIndices[k] = k;
		Vector<int> vIndex = new (laneIndices);
		Vector<int> vSteps = new (_simdWidth);
			
		Vector<int> vMaxIndex = Vector<int>.Zero;
		Vector<float> vMax = new (float.NegativeInfinity);

		int i = 0;
		for (; i <= a.Length - _simdWidth; i += _simdWidth) {
			Vector<float> vValues = new (span.Slice(i, _simdWidth));

			Vector<int> vMask = Vector.GreaterThan(vValues, vMax);
				
			vMax = Vector.ConditionalSelect(vMask, vValues, vMax);
			vMaxIndex = Vector.ConditionalSelect(vMask, vIndex, vMaxIndex);
				
			vIndex += vSteps;
		}
			

		int maxIndex = vMaxIndex[0];
		float max = vMax[0];
		for (int j = 1; j < _simdWidth; j++) { 
			if (vMax[j] > max) {
				max = vMax[j];
				maxIndex = vMaxIndex[j];
			}
		}

		for (; i < a.Length; i++) {
			if (span[i] > max) {
				max = span[i];
				maxIndex = i;
			}
		}

		return (maxIndex, max);
	}
}