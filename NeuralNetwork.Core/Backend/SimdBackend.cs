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
		Console.WriteLine($"Using Simd Backend with width {_simdWidth} floats. ({_simdWidth * sizeof(float) * 8} bits)");
	}

	private (int, float) MaxVectorIndex(CpuTensor<float> a) {
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

	protected override void CpuMatrixTranspose(CpuTensor<float> a, CpuTensor<float> output) {
		for (int row = 0; row < a.Shape[0]; row++) {
			for (int col = 0; col < a.Shape[1]; col++) {
				output[col, row] = a[row, col];
			}
		}
	}

	protected override void CpuLoss(CpuTensor<float> activation, CpuValue<int> label, CpuValue<float> lossOut) {
		const float epsilon = 1e-15f;
		lossOut.Value = -(float)Math.Log(activation[label] + epsilon);
	}

	protected override void CpuInput(CpuTensor<float> input, CpuTensor<float> activationOut) {
		const float mean = 0.1307f;
		const float std = 0.3081f;

		Vector<float> vMean = new (mean);
		Vector<float> vStd = new (std);
			
		ReadOnlySpan<float> inputSpan = input.Data;
			
		Span<float> activationOutSpan = activationOut.Data;

		int i = 0;
		for (; i <= input.Length - _simdWidth; i += _simdWidth) {
			Vector<float> vInput = new (inputSpan.Slice(i, _simdWidth));
			((vInput - vMean) / vStd).CopyTo(activationOutSpan.Slice(i, _simdWidth));
		}

		for (; i < input.Length; i++) {
			activationOutSpan[i] = (input[i] - mean) / std;
		}
	}

	protected override void CpuForwardPass(CpuTensor<float> input, CpuTensor<float> weights, CpuTensor<float> bias, CpuTensor<float> valuesOut) {
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
				Vector<float> vWeights = new (weightsSpan.Slice(offset + col, _simdWidth));
				Vector<float> vInput = new (inputSpan.Slice(col, _simdWidth));
				vSum += vWeights * vInput;
			}
				
			float sum = biasSpan[row] + Vector.Sum(vSum);

			for (; col < columns; col++) {
				sum += weightsSpan[offset + col] * inputSpan[col];
			}

			valuesOutSpan[row] = sum;
		}
	}

	protected override void CpuActivation(CpuTensor<float> values, CpuTensor<float> activationOut) {
		ReadOnlySpan<float> valuesSpan = values.Data;
			
		Span<float> activationOutSpan = activationOut.Data;
			
		Vector<float> vZero = Vector<float>.Zero;

		int i = 0;
		for (; i <= values.Length - _simdWidth; i += _simdWidth) {
			Vector<float> vValues = new (valuesSpan.Slice(i, _simdWidth));
			Vector.Max(vValues, vZero).CopyTo(activationOutSpan.Slice(i, _simdWidth));
		}

		for (; i < values.Length; i++) {
			activationOutSpan[i] = valuesSpan[i] > 0 ?  valuesSpan[i] : 0;
		}
	}

	protected override void CpuOutput(CpuTensor<float> values, CpuTensor<float> activationOut, CpuValue<int> predictionOut) {
		ReadOnlySpan<float> valuesSpan = values.Data;
			
		Span<float> activationOutSpan = activationOut.Data;
			
		// Obtaining Max

		(int maxIndex, float max) = MaxVectorIndex(values);

		// Creating each e^(...) value and the sum of them all
		Vector<float> vMax =  new Vector<float>(max);
		Vector<float> vSum = Vector<float>.Zero;
			
		int i;
		for (i = 0; i <= values.Length - _simdWidth; i += _simdWidth) {
			Vector<float> vValues = new (valuesSpan.Slice(i, _simdWidth));
			Vector<float> vE = Vector.Exp(vValues - vMax);
				
			vE.CopyTo(activationOutSpan.Slice(i, _simdWidth));
			vSum += vE;
		}

		float sum = Vector.Sum(vSum);
			
		for (; i < values.Length; i++) {
			float e = (float)Math.Exp(valuesSpan[i] - max);
			activationOutSpan[i] = e;
			sum += e;
		}

		// Divide the final output with the sum
			
		for (i = 0; i < activationOut.Length - _simdWidth; i += _simdWidth) {
			Vector<float> vActivation = new (activationOutSpan.Slice(i, _simdWidth));
			(vActivation / sum).CopyTo(activationOutSpan.Slice(i, _simdWidth));
		}

		for (; i < activationOut.Length; i++) {
			activationOutSpan[i] /= sum;
		}

		predictionOut.Value = maxIndex;
	}

	protected override void CpuBackward(CpuTensor<float> values, CpuTensor<float> delta, CpuTensor<float> weightsT, CpuTensor<float> deltaOut) {
		int rows = weightsT.Shape[0];
		int columns = weightsT.Shape[1];
			
		ReadOnlySpan<float> valuesSpan = values.Data;
		ReadOnlySpan<float> deltaSpan = delta.Data;
		ReadOnlySpan<float> weightsTSpan = weightsT.Data;
			
		Span<float> deltaOutSpan = deltaOut.Data;

		for (int row = 0; row < rows; row++) {
			if (valuesSpan[row] <= 0) {
				deltaOutSpan[row] = 0f;
				continue;
			}
			
			int offset = row * columns;

			Vector<float> vSum = Vector<float>.Zero;
				
			int col = 0;
			for (; col <= columns - _simdWidth; col += _simdWidth) {
				Vector<float> vWeightsT = new(weightsTSpan.Slice(offset + col, _simdWidth));
				Vector<float> vDelta = new(deltaSpan.Slice(col, _simdWidth));
				vSum += vWeightsT * vDelta;
			}
				
			float sum = Vector.Sum(vSum);

			for (; col < columns; col++) {
				sum += weightsTSpan[offset + col] * deltaSpan[col];
			}

			deltaOutSpan[row] = sum;
		}
	}

	protected override void CpuBackwardOutput(CpuTensor<float> activations, CpuTensor<float> expectedValues, CpuTensor<float> deltaOut) {
		int rows = activations.Shape[0];
			
		ReadOnlySpan<float> activationsSpan = activations.Data;
		ReadOnlySpan<float> expectedValuesSpan = expectedValues.Data;
			
		Span<float> deltaOutSpan = deltaOut.Data;

		int row = 0;
		for (; row <= rows - _simdWidth; row += _simdWidth) {
			Vector<float> vActivation = new (activationsSpan.Slice(row, _simdWidth));
			Vector<float> vExpectedValues = new (expectedValuesSpan.Slice(row, _simdWidth));
				
			(vActivation - vExpectedValues).CopyTo(deltaOutSpan.Slice(row, _simdWidth));
		}

		for (; row < activationsSpan.Length; row++) {
			deltaOutSpan[row] = activationsSpan[row] - expectedValuesSpan[row];
		}
	}

	protected override void CpuBackPropagation(CpuTensor<float> delta, CpuTensor<float> activations, CpuTensor<float> weightDelta, CpuTensor<float> biasDelta) {
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
				Vector<float> vActivation = new (activationsSpan.Slice(col, _simdWidth));
				Vector<float> vWeightsDelta = new (weightDeltaSpan.Slice(offset + col, _simdWidth));
					
				(vActivation * deltaSpan[row] + vWeightsDelta).CopyTo(weightDeltaSpan.Slice(offset + col, _simdWidth));
			}

			for (; col < columns; col++) {
				weightDeltaSpan[offset + col] += deltaSpan[row] * activationsSpan[col];
			}
		}

		row = 0;
		for (; row < rows - _simdWidth; row += _simdWidth) {
			Vector<float> vDelta = new (deltaSpan.Slice(row, _simdWidth));
			Vector<float> vBiasDelta = new (biasDeltaSpan.Slice(row, _simdWidth));
				
			(vBiasDelta + vDelta).CopyTo(biasDeltaSpan.Slice(row, _simdWidth));
		}

		for (; row < rows; row++) {
			biasDeltaSpan[row] += deltaSpan[row];
		}
	}

	protected override void CpuAdjust(CpuTensor<float> weightDelta, CpuTensor<float> biasDelta, float scale, CpuTensor<float> weights, CpuTensor<float> weightsT, CpuTensor<float> bias) {
		ReadOnlySpan<float> weightDeltaSpan = weightDelta.Data;
		ReadOnlySpan<float> biasDeltaSpan = biasDelta.Data;
			
		Span<float> weightsSpan = weights.Data;
		Span<float> biasSpan = bias.Data;
			
		int length = weights.Length;
		int i = 0;
		for (; i <= length - _simdWidth; i += _simdWidth) {
			Vector<float> vWeights = new (weightsSpan.Slice(i, _simdWidth));
			Vector<float> vWeightDelta = new (weightDeltaSpan.Slice(i, _simdWidth));
			(vWeights - vWeightDelta * scale).CopyTo(weightsSpan.Slice(i, _simdWidth));
		}

		for (; i < length; i++) {
			weightsSpan[i] -= weightDeltaSpan[i] * scale;
		}
			
		CpuMatrixTranspose(weights, weightsT);

		length = bias.Length;
		i = 0;
		for (; i <= length - _simdWidth; i += _simdWidth) {
			Vector<float> vBias = new (biasSpan.Slice(i, _simdWidth));
			Vector<float> vBiasDelta = new (biasDeltaSpan.Slice(i, _simdWidth));
			(vBias - vBiasDelta * scale).CopyTo(biasSpan.Slice(i, _simdWidth));
		}

		for (; i < length; i++) {
			biasSpan[i] -= biasDeltaSpan[i] * scale;
		}
	}
}
	