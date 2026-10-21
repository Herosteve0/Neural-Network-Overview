using NeuralNetwork.Core.System.Data;

namespace NeuralNetwork.Core.System.Backend;

/**
 * Mathematically correct backend.
 */
public class ScalarBackend : CpuBackend {
	public ScalarBackend() {
		Console.WriteLine("Using Scalar Backend.");
	}
	
	protected override void CpuMatrixTranspose(CpuTensor a, CpuTensor output) {
		for (int row = 0; row < a.Shape[0]; row++) {
			for (int col = 0; col < a.Shape[1]; col++) {
				output[col, row] = a[row, col];
			}
		}
	}

	protected override void CpuLoss(CpuTensor activation, int label, CpuValue lossOut) {
		const float epsilon = 1e-15f;
		lossOut.Value = -(float)Math.Log(activation[label] + epsilon);
	}

	protected override void CpuInput(CpuTensor input, CpuTensor activationOut) {
		const float mean = 0.1307f;
		const float std = 0.3081f;

		int length = input.Shape[0];

		for (int i = 0; i < length; i++) {
			activationOut[i] = (input[i] - mean) / std;
		}
	}
		
	protected override void CpuForwardPass(CpuTensor input, CpuTensor weights, CpuTensor bias, CpuTensor valuesOut) {
		int lengthRows = weights.Shape[0];
		int lengthColumns = weights.Shape[1];
		
		for (int row = 0; row < lengthRows; row++) {
			valuesOut[row] = bias[row];
			for (int col = 0; col < lengthColumns; col++) {
				valuesOut[row] += weights[row, col] * input[col];
			}
		}
	}
	protected override void CpuActivation(CpuTensor values, CpuTensor activationOut) {
		int length = values.Shape[0];
		
		for (int i = 0; i < length; i++) {
			activationOut[i] = Math.Max(values[i], 0);
		}
	}
	protected override void CpuOutput(CpuTensor values, CpuTensor activationOut, CpuInt predictionOut) {
		int length = values.Shape[0];
		
		int maxIndex = 0;
		for (int i = 1; i < length; i++) {
			if (values[maxIndex] < values[i]) {
				maxIndex = i;
			}
		}
		float max = values[maxIndex];

		float sum = 0f;
		for (int i = 0; i < length; i++) {
			float e = (float)Math.Exp(values[i] - max);
			activationOut[i] = e;
			sum += e;
		}

		for (int i = 0; i < length; i++) {
			activationOut[i] /= sum;
		}
		
		predictionOut.Value = maxIndex;
	}

	protected override void CpuBackward(CpuTensor values, CpuTensor delta, CpuTensor weightsT, CpuTensor deltaOut) {
		int lengthRows = weightsT.Shape[0];
		int lengthColumns = weightsT.Shape[1];

		for (int row = 0; row < lengthRows; row++) {
			if (values[row] <= 0) {
				deltaOut[row] = 0f;
				continue;
			}
			deltaOut[row] = 0f;
			for (int col = 0; col < lengthColumns; col++) {
				deltaOut[row] += weightsT[row, col] * delta[col];
			}
		}
	}
	protected override void CpuBackwardOutput(CpuTensor activations, CpuTensor expectedValues, CpuTensor deltaOut) {
		int length = activations.Shape[0];
		
		for (int i = 0; i < length; i++) {
			deltaOut[i] = activations[i] - expectedValues[i];
		}
	}
	protected override void CpuBackPropagation(CpuTensor delta, CpuTensor activations, CpuTensor weightDelta, CpuTensor biasDelta) {
		int lengthRows = delta.Shape[0];
		int lengthColumns = activations.Shape[0];
		
		for (int row = 0; row < lengthRows; row++) {
			for (int col = 0; col < lengthColumns; col++) {
				weightDelta[row, col] += delta[row] * activations[col];
			}
			biasDelta[row] += delta[row];
		}
	}

	protected override void CpuAdjust(CpuTensor weightDelta, CpuTensor biasDelta, float scale, CpuTensor weights, CpuTensor weightsT, CpuTensor bias) {
		int lengthRows = weightDelta.Shape[0];
		int lengthColumns = weightDelta.Shape[1];

		for (int row = 0; row < lengthRows; row++) {
			for (int col = 0; col < lengthColumns; col++) {
				weights[row, col] -= weightDelta[row, col] * scale;
				weightsT[col, row] = weights[row, col];
			}
			bias[row] -= biasDelta[row] * scale;
		}
	}
}
