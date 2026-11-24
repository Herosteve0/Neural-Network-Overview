using NeuralNetwork.Core.Data.SingleValues;
using NeuralNetwork.Core.Data.Tensors;

namespace NeuralNetwork.Core.Backend;

/**
 * Scalar backend. Slowest by an embarrassing amount.
 */
public class ScalarBackend : CpuBackend {
	public ScalarBackend() {
		Console.WriteLine("Initialized Scalar Backend.");
	}
	
	protected override void CpuInput(CpuTensor<float> input, CpuTensor<float> activationOut) {
		const float mean = 0.1307f;
		const float std = 0.3081f;

		int length = input.Shape[0];

		for (int i = 0; i < length; i++) {
			activationOut[i] = (input[i] - mean) / std;
		}
	}

	protected override void CpuMaxIndex(CpuTensor<float> input, CpuValue<int> index) {
		int length = input.Shape[0];

		index.Value = 0;
		for (int i = 1; i < length; i++) {
			if (input[i] > input[index]) {
				index.Value = i;
			}
		}
	}

	protected override void CpuLinear(
		CpuTensor<float> input, CpuTensor<float> weights, CpuTensor<float> bias, CpuTensor<float> valuesOut
	) {
		int lengthRows = weights.Shape[0];
		int lengthColumns = weights.Shape[1];

		for (int row = 0; row < lengthRows; row++) {
			valuesOut[row] = bias[row];
			for (int col = 0; col < lengthColumns; col++) {
				valuesOut[row] += weights[row, col] * input[col];
			}
		}
	}

	protected override void CpuDelta(CpuTensor<float> delta, CpuTensor<float> weights, CpuTensor<float> deltaOut) {
		int lengthRows = weights.Shape[1];
		int lengthColumns = weights.Shape[0];

		for (int row = 0; row < lengthRows; row++) {
			deltaOut[row] = 0f;
			for (int col = 0; col < lengthColumns; col++) {
				deltaOut[row] += weights[col, row] * delta[col];
			}
		}
	}

	protected override void CpuBackPropagation(
		CpuTensor<float> delta, CpuTensor<float> activations, CpuTensor<float> weightDelta, CpuTensor<float> biasDelta
	) {
		int lengthRows = delta.Shape[0];
		int lengthColumns = activations.Shape[0];

		for (int row = 0; row < lengthRows; row++) {
			for (int col = 0; col < lengthColumns; col++) {
				weightDelta[row, col] += delta[row] * activations[col];
			}

			biasDelta[row] += delta[row];
		}
	}

	protected override void CpuAdjust(
		CpuTensor<float> weightDelta, CpuTensor<float> biasDelta, float scale,
		CpuTensor<float> weights, CpuTensor<float> bias
	) {
		int lengthRows = weightDelta.Shape[0];
		int lengthColumns = weightDelta.Shape[1];

		for (int row = 0; row < lengthRows; row++) {
			for (int col = 0; col < lengthColumns; col++) {
				weights[row, col] -= weightDelta[row, col] * scale;
			}

			bias[row] -= biasDelta[row] * scale;
		}
	}

	protected override void CpuDropout(CpuTensor<float> input, CpuTensor<float> mask, CpuTensor<float> output) {
		int length = input.Length;
		
		for (int i = 0; i < length; i++) {
			output[i] = input[i] * mask[i];
		}
	}


	protected override void CpuReLuForward(CpuTensor<float> input, CpuTensor<float> output) {
		int length = input.Length;
		
		for (int i = 0; i < length; i++) {
			output[i] = Math.Max(input[i], 0);
		}
	}
	protected override void CpuReLuBackward(CpuTensor<float> input, CpuTensor<float> gradient) {
		int length = input.Length;
		
		for (int i = 0; i < length; i++) {
			if (input[i] >= 0) continue;
			gradient[i] = 0;
		}
	}

	protected override void CpuLeakyReLuForward(CpuTensor<float> input, float a, CpuTensor<float> output) {
		int length = input.Length;
		
		for (int i = 0; i < length; i++) {
			output[i] = input[i] * (input[i] > 0 ? 1f : a);
		}
	}
	protected override void CpuLeakyReLuBackward(CpuTensor<float> input, float a, CpuTensor<float> gradient) {
		int length = input.Length;
		
		for (int i = 0; i < length; i++) {
			gradient[i] *= input[i] > 0 ? 1f : a;
		}
	}

	protected override void CpuSoftmaxForward(CpuTensor<float> input, CpuTensor<float> output,
		CpuValue<int> predictionOut) {
		int length = input.Length;
		
		int maxIndex = 0;
		for (int i = 1; i < length; i++) {
			if (input[maxIndex] < input[i]) {
				maxIndex = i;
			}
		}
		float max = input[maxIndex];

		float sum = 0f;
		for (int i = 0; i < length; i++) {
			float e = (float)Math.Exp(input[i] - max);
			output[i] = e;
			sum += e;
		}

		for (int i = 0; i < length; i++) {
			output[i] /= sum;
		}

		predictionOut?.Value = maxIndex;
	}

	protected override void CpuSoftmaxBackward(CpuTensor<float> input, CpuTensor<float> gradient) {
		int length = input.Length;

		float max = input[0];
		for (int i = 1; i < length; i++) {
			max = Math.Max(input[i], max);
		}

		float sum = 0f;
		for (int i = 0; i < length; i++) {
			sum += (float)Math.Exp(input[i]);
		}

		float dot = 0f;
		for (int i = 0; i < length; i++) {
			float probability = (float)Math.Exp(input[i] - max) / sum;
			dot += gradient[i] * probability;
		}

		for (int i = 0; i < length; i++) {
			float probability = (float)Math.Exp(input[i] - max) / sum;
			gradient[i] = probability * (gradient[i] - dot);
		}
	}

	protected override void CpuSigmoidForward(CpuTensor<float> input, CpuTensor<float> output) {
		int length = input.Length;

		for (int i = 0; i < length; i++) {
			float e = (float)Math.Exp(-input[i]);
			output[i] = 1 / (1 + e);
		}
	}
	protected override void CpuSigmoidBackward(CpuTensor<float> input, CpuTensor<float> gradient) {
		int length = input.Length;

		for (int i = 0; i < length; i++) {
			float e = (float)Math.Exp(-Math.Abs(input[i]));
			gradient[i] *= e / ((1 + e) * (1 + e));
		}
	}


	protected override void CpuMseCalculation(
		CpuTensor<float> activation, CpuValue<int> label, float a, CpuValue<float> lossOut
	) {
		int length = activation.Length;

		lossOut.Value = 0;
		for (int i = 0; i < length; i++) {
			float value = i == label ? 1 : 0;
			lossOut.Value += (activation[i] - value) * (activation[i] - value);
		}

		lossOut.Value *= a;
	}

	protected override void CpuMseBackward(
		CpuTensor<float> input, CpuTensor<float> expectedInput, float a, CpuTensor<float> output
	) {
		int length = input.Length;

		for (int i = 0; i < length; i++) {
			output[i] = 2 * a * (input[i] - expectedInput[i]);
		}
	}

	protected override void CpuCrossEntropyCalculate(
		CpuTensor<float> activation, CpuValue<int> label, CpuValue<float> lossOut
	) {
		const float epsilon = 1e-15f;
		lossOut.Value = -(float)Math.Log(activation[label] + epsilon);
	}

	protected override void CpuCrossEntropyBackward(
		CpuTensor<float> input, CpuTensor<float> expectedInput, CpuTensor<float> output
	) {
		int length = input.Length;

		for (int i = 0; i < length; i++) {
			output[i] = input[i] - expectedInput[i];
		}
	}
}
