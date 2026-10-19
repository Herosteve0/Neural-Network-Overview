using System.Numerics;
using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using NeuralNetwork.Core.System.Data;

// ReSharper disable InvalidXmlDocComment

namespace NeuralNetwork.Core.System;

using FloatView = ArrayView1D<float, Stride1D.Dense>;
using FloatViewSingle = ArrayView1D<float, Stride1D.Dense>;

public interface IBackend {
	/**
	 * Creates an appropriate tensor.
	 */
	ITensor CreateTensor(params int[] shape);
	/**
	 * Creates an appropriate tensor with the given data.
	 * Data is copied.
	 */
	ITensor CreateTensor(float[] data, params int[] shape);
		
	/**
	 * Creates an appropriate variable.
	 */
	ISingleValue CreateValue();
	/**
	 * Creates an appropriate variable with the given data.
	 */
	ISingleValue CreateValue(float data);
		
	/**
	 * Creates an appropriate tensor filled with Zeros.
	 */
	ITensor Zero(params int[] shape);
	/**
	 * Creates an appropriate randomized tensor.
	 */
	ITensor Random(Random random, float min, float max, params int[] shape);

	/**
	 * Sets the tensor values to a given array of floats.
	 */
	void Set(ITensor a, float[] data);
	/**
	 * Converts the tensor into an array of floats.
	 */
	float[] Read(ITensor a);
	/**
	 * Sets the value to a given float.
	 */
	void SetSingle(ISingleValue a, float data);
	/**
	 * Converts the value to a float.
	*/
	float ReadSingle(ISingleValue a);

	/**
	 * Adds a value to the first value.
	 */
	void AddSingle(ISingleValue value, ISingleValue change);

	/**
	 * Checks whether the prediction was correct and increases a given accuracy value if so.
	 */
	public void CheckAnswer(ISingleValue prediction, ISingleValue label, ISingleValue accuracyOut);
	
	/**
	 * Returns a Transposed Matrix.
	 */
	void MatrixTranspose(ITensor a, ITensor output);

	/**
	 * Returns the Loss
	 */
	void Loss(ITensor activation, int label, ISingleValue lossOut);
	
	/**
	 * The calculations for Activations of the first layer.
	 */
	void Input(ITensor input, ITensor activationOut);

	/**
	 * The calculation for the Values of a Layer.
	 */
	void ForwardPass(ITensor input, ITensor weights, ITensor bias, ITensor valuesOut);
		
	/**
	 * The calculations for the Activations of a Layer.
	 */
	void Activation(ITensor values, ITensor activationOut);
	/**
	 * The calculations for the Output of the last layer.
	 */
	void Output(ITensor values, ITensor activationOut, ISingleValue predictionOut);

	/**
	 * The calculations for the Delta value that will be applied to the Weights and Biases.
	 */
	void Backward(ITensor values, ITensor delta, ITensor weightsT, ITensor deltaOut);
	/**
	 * The calculations for the initial Delta value at the output Layer.
	 */
	void BackwardOutput(ITensor activations, ITensor expectedValues, ITensor deltaOut);
	/**
	 * The calculations for the change that will be applied to the Weights and Biases.
	 */
	void BackPropagation(ITensor delta, ITensor activations, ITensor weightDelta, ITensor biasDelta);
		
	/**
	 * Adjusts the Weights and Biases in proportion with the calculated Delta.
	 */
	void Adjust(ITensor weightDelta, ITensor biasDelta, float scale, ITensor weights, ITensor weightsT, ITensor bias);

	void Synchronize();
}
	
// CPU

public abstract class CpuBackend : IBackend {
	public ITensor CreateTensor(params int[] shape) {
		return new CpuTensor(shape);
	}
	public ITensor CreateTensor(float[] data, params int[] shape) {
		return new CpuTensor(data, shape);
	}
	public ISingleValue CreateValue() {
		return new CpuValue();
	}
	public ISingleValue CreateValue(float data) {
		return new CpuValue(data);
	}
	public ITensor Zero(params int[] shape) {
		return CpuTensor.Zero(shape);
	}
	public ITensor Random(Random random, float min, float max, params int[] shape) {
		return CpuTensor.Random(random, min, max, shape);
	}

	public void Set(ITensor a, float[] data) {
		((CpuTensor)a).Data = data;
	}
	public float[] Read(ITensor a) {
		return ((CpuTensor)a).Data.ToArray();
	}
	public void SetSingle(ISingleValue a, float data) {
		((CpuValue)a).Value = data;
	}
	public float ReadSingle(ISingleValue a) {
		return ((CpuValue)a).Value;
	}

	public void AddSingle(ISingleValue value, ISingleValue change) {
		((CpuValue)value).Value += (CpuValue)change;
	}

	public void MatrixTranspose(ITensor a, ITensor output) {
		CpuMatrixTranspose((CpuTensor)a, (CpuTensor)output);
	}

	public void CheckAnswer(ISingleValue prediction, ISingleValue label, ISingleValue accuracyOut) {
		if ((int)((CpuValue)prediction).Value == (int)((CpuValue)label).Value) ((CpuValue)accuracyOut).Value++;
	}
	
	public void Loss(ITensor activation, int label, ISingleValue lossOut) {
		CpuLoss((CpuTensor)activation, label, (CpuValue)lossOut);
	}

	public void Input(ITensor input, ITensor activationOut) {
		CpuInput((CpuTensor)input, (CpuTensor)activationOut);
	}

	public void ForwardPass(ITensor input, ITensor weights, ITensor bias, ITensor valuesOut) {
		CpuForwardPass((CpuTensor)input, (CpuTensor)weights, (CpuTensor)bias, (CpuTensor)valuesOut);
	}
	public void Activation(ITensor values, ITensor activationOut) {
		CpuActivation((CpuTensor)values, (CpuTensor)activationOut);
	}
	public void Output(ITensor values, ITensor activationOut, ISingleValue predictionOut) {
		CpuOutput((CpuTensor)values, (CpuTensor)activationOut, (CpuValue)predictionOut);
	}

	public void Backward(ITensor values, ITensor delta, ITensor weightsT, ITensor deltaOut) {
		CpuBackward((CpuTensor)values, (CpuTensor)delta, (CpuTensor)weightsT, (CpuTensor)deltaOut);
	}

	public void BackwardOutput(ITensor activations, ITensor expectedValues, ITensor deltaOut) {
		CpuBackwardOutput((CpuTensor)activations, (CpuTensor)expectedValues, (CpuTensor)deltaOut);
	}

	public void BackPropagation(ITensor delta, ITensor activations, ITensor weightDelta, ITensor biasDelta) {
		CpuBackPropagation((CpuTensor)delta, (CpuTensor)activations, (CpuTensor)weightDelta, (CpuTensor)biasDelta);
	}

	public void Adjust(ITensor weightDelta, ITensor biasDelta, float scale, ITensor weights, ITensor weightsT, ITensor bias) {
		CpuAdjust((CpuTensor)weightDelta, (CpuTensor)biasDelta, scale, (CpuTensor)weights, (CpuTensor)weightsT, (CpuTensor)bias);
	}

	public void Synchronize() { }

	protected abstract void CpuMatrixTranspose(CpuTensor a, CpuTensor output);

	protected abstract void CpuLoss(CpuTensor activation, int label, CpuValue lossOut);
	
	protected abstract void CpuInput(CpuTensor input, CpuTensor activationOut);
		
	protected abstract void CpuForwardPass(CpuTensor input, CpuTensor weights, CpuTensor bias, CpuTensor valuesOut);
	protected abstract void CpuActivation(CpuTensor values, CpuTensor activationOut);
	protected abstract void CpuOutput(CpuTensor values, CpuTensor activationOut, CpuValue predictionOut);
		
	protected abstract void CpuBackward(CpuTensor values, CpuTensor delta, CpuTensor weightsT, CpuTensor deltaOut);
	protected abstract void CpuBackwardOutput(CpuTensor activations, CpuTensor expectedValues, CpuTensor deltaOut);
	protected abstract void CpuBackPropagation(CpuTensor delta, CpuTensor activations, CpuTensor weightDelta, CpuTensor biasDelta);
		
	protected abstract void CpuAdjust(CpuTensor weightDelta, CpuTensor biasDelta, float scale, CpuTensor weights, CpuTensor weightsT, CpuTensor bias);
}

/**
 * Mathematically correct backend.
 */
public class ScalarBackend : CpuBackend {
	public ScalarBackend() {
		Console.WriteLine("Using Scalar Backend.");
	}
		
	private static void Addition(CpuTensor a, CpuTensor b, CpuTensor output) {
		for (int i = 0; i < output.Length; i++) {
			output.Data[i] = a.Data[i] + b.Data[i];
		}
	}
	private static void Subtraction(CpuTensor a, CpuTensor b, CpuTensor output) {
		for (int i = 0; i < output.Length; i++) {
			output.Data[i] = a.Data[i] - b.Data[i];
		}
	}
	private static void Scale(CpuTensor a, float scalar, CpuTensor output) {
		for (int i = 0; i < output.Length; i++) {
			output.Data[i] = a.Data[i] * scalar;
		}
	}
	private static void MatrixVectorMultiplication(CpuTensor a, CpuTensor b, CpuTensor output) {
		for (int row = 0; row < a.Shape[0]; row++) {
			float sum = 0;
			for (int col = 0; col < a.Shape[1]; col++) {
				sum += a[row, col] * b[col];
			}
			output[row] = sum;
		}
	}
	private static void VectorVectorMultiplication(CpuTensor a, CpuTensor b, CpuTensor output) {
		for (int row = 0; row < a.Shape[0]; row++) {
			for (int col = 0; col < b.Shape[0]; col++) {
				output[row, col] = a[row] * b[col];
			}
		}
	}
	private static void ElementMultiplication(CpuTensor a, CpuTensor b, CpuTensor output) {
		for (int i = 0; i < a.Length; i++) {
			output.Data[i] = a.Data[i] * b.Data[i];
		}
	}
	private static void VectorNormalize(CpuTensor a, CpuTensor output) {
		const float mean = 0.1307f;
		const float std = 0.3081f;

		for (int i = 0; i < a.Shape[0]; i++) {
			output[i] = (a[i] - mean) / std;
		}
	}
	private static void VectorReLu(CpuTensor a, CpuTensor output) {
		for (int i = 0; i < a.Shape[0]; i++) {
			output[i] = a[i] > 0 ? a[i] : 0;
		}
	}
	private static void VectorReLuDerivative(CpuTensor a, CpuTensor output) {
		for (int i = 0; i < a.Shape[0]; i++) {
			output[i] = a[i] > 0 ? 1 : 0;
		}
	}
	private static void VectorSoftMax(CpuTensor a, CpuTensor output) {
		float max = a[0];
		for (int i = 1; i < a.Shape[0]; i++) {
			if (max < a[i]) max = a[i];
		}

		float sum = 0f;
		for (int i = 0; i < a.Shape[0]; i++) {
			float e = (float)Math.Exp(a[i] - max);
			output[i] = e;
			sum += e;
		}

		for (int i = 0; i < a.Shape[0]; i++) {
			output[i] /= sum;
		}
	}

	protected override void CpuMatrixTranspose(CpuTensor a, CpuTensor output) {
		for (int row = 0; row < a.Shape[0]; row++) {
			for (int col = 0; col < a.Shape[1]; col++) {
				output[col, row] = a[row, col];
			}
		}
	}

	protected override void CpuLoss(CpuTensor activation, int label, CpuValue lossOut) {
		lossOut.Value = -(float)Math.Log(activation[label]);
	}

	protected override void CpuInput(CpuTensor input, CpuTensor activationOut) {
		/*
		 * InputNormalizing(Input)
		 */
		VectorNormalize(input, activationOut);
	}
		
	protected override void CpuForwardPass(CpuTensor input, CpuTensor weights, CpuTensor bias, CpuTensor valuesOut) {
		CpuTensor temp = new CpuTensor(weights.Shape[0]);

		/*
		 * Weights * Input
		 */
		MatrixVectorMultiplication(weights, input, temp);
			
		/*
		 * (Weights * Input) + Bias
		 */
		Addition(temp, bias, valuesOut);
	}
	protected override void CpuActivation(CpuTensor values, CpuTensor activationOut) {
		/*
		 * ActivationFunc(Value)
		 */
		VectorReLu(values, activationOut);
	}
	protected override void CpuOutput(CpuTensor values, CpuTensor activationOut, CpuValue predictionOut) {
		/*
		 * OutputFunc(Value)
		 */
		VectorSoftMax(values, activationOut);
			
		int maxIndex = 0;
		for (int i = 1; i < activationOut.Shape[0]; i++) {
			if (activationOut[i] > activationOut[maxIndex]) maxIndex = i;
		}
		predictionOut.Value = maxIndex;
	}

	protected override void CpuBackward(CpuTensor values, CpuTensor delta, CpuTensor weightsT, CpuTensor deltaOut) {
		CpuTensor temp = new CpuTensor(weightsT.Shape[0]);
			
		/*
		 * Weights^T * Delta
		 */
		MatrixVectorMultiplication(weightsT, delta, temp);

		CpuTensor tempDer = new CpuTensor(weightsT.Shape[0]);
			
		/*
		 *
		 */
		VectorReLuDerivative(values, tempDer);
			
		/*
		 * (Weights^T * Delta) ● ActivationFunc'(Value)
		 */
		ElementMultiplication(temp, tempDer, deltaOut);

	}
	protected override void CpuBackwardOutput(CpuTensor activations, CpuTensor expectedValues, CpuTensor deltaOut) {
			
		/*
		 * Activation - CorrectValues
		 */
			
		Subtraction(activations, expectedValues, deltaOut);
	}
	protected override void CpuBackPropagation(CpuTensor delta, CpuTensor activations, CpuTensor weightDelta, CpuTensor biasDelta) {
		CpuTensor temp = new CpuTensor(delta.Shape[0], activations.Shape[0]);
			
		/*
		 * Delta * Activations
		 */
		VectorVectorMultiplication(delta, activations, temp);
			
		/*
		 * WeightsDelta += Delta * Activations
		 */
		Addition(weightDelta, temp, weightDelta);
			
		/*
		 * BiasDelta += Delta
		 */
		Addition(biasDelta, delta, biasDelta);
	}

	protected override void CpuAdjust(CpuTensor weightDelta, CpuTensor biasDelta, float scale, CpuTensor weights, CpuTensor weightsT, CpuTensor bias) {
		CpuTensor temp = new CpuTensor(weights.Shape[0], weights.Shape[1]);
		/*
		 * WeightDelta * scale
		 */
		Scale(weightDelta, scale, temp);
		/*
		 * Weights -= WeightsDelta * scale
		 */
		Subtraction(weights, temp, weights);
			
		CpuMatrixTranspose(weights, weightsT);

		temp = new CpuTensor(bias.Shape[0]);
		/*
		 * BiasDelta * scale
		 */
		Scale(biasDelta, scale, temp);
		/*
		 * Bias -= BiasDelta * scale
		 */
		Subtraction(bias, temp, bias);
	}
}

/**
 * Cpu Simd powered backend.
 */
public class SimdBackend : CpuBackend {
	private readonly int _simdWidth = Vector<float>.Count;

	public SimdBackend() {
		Console.WriteLine($"Using Simd Backend with width {_simdWidth}.");
	}

	private (int, float) MaxVectorIndex(CpuTensor a) {
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

	protected override void CpuMatrixTranspose(CpuTensor a, CpuTensor output) {
		for (int row = 0; row < a.Shape[0]; row++) {
			for (int col = 0; col < a.Shape[1]; col++) {
				output[col, row] = a[row, col];
			}
		}
	}

	protected override void CpuLoss(CpuTensor activation, int label, CpuValue lossOut) {
		lossOut.Value = -(float)Math.Log(activation[label]);
	}

	protected override void CpuInput(CpuTensor input, CpuTensor activationOut) {
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

	protected override void CpuForwardPass(CpuTensor input, CpuTensor weights, CpuTensor bias, CpuTensor valuesOut) {
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

	protected override void CpuActivation(CpuTensor values, CpuTensor activationOut) {
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

	protected override void CpuOutput(CpuTensor values, CpuTensor activationOut, CpuValue predictionOut) {
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

	protected override void CpuBackward(CpuTensor values, CpuTensor delta, CpuTensor weightsT, CpuTensor deltaOut) {
		int rows = weightsT.Shape[0];
		int columns = weightsT.Shape[1];
			
		ReadOnlySpan<float> valuesSpan = values.Data;
		ReadOnlySpan<float> deltaSpan = delta.Data;
		ReadOnlySpan<float> weightsTSpan = weightsT.Data;
			
		Span<float> deltaOutSpan = deltaOut.Data;

		for (int row = 0; row < rows; row++) {
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

			deltaOutSpan[row] = valuesSpan[row] > 0 ? sum : 0;
		}
	}

	protected override void CpuBackwardOutput(CpuTensor activations, CpuTensor expectedValues, CpuTensor deltaOut) {
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

	protected override void CpuBackPropagation(CpuTensor delta, CpuTensor activations, CpuTensor weightDelta, CpuTensor biasDelta) {
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

	protected override void CpuAdjust(CpuTensor weightDelta, CpuTensor biasDelta, float scale, CpuTensor weights, CpuTensor weightsT, CpuTensor bias) {
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
			biasDelta[i] -= biasDeltaSpan[i] * scale;
		}
	}
}
	
/**
 * Gpu powered backend.
 */
public class GpuBackend : IBackend {

	private readonly Accelerator _accelerator;

	private readonly ArithmeticKernels _arithmeticKernels;
	private readonly MatrixKernels _matrixKernels;
	private readonly ReductionKernels _reductionKernels;
	private readonly FunctionKernels _functionKernels;
	private readonly MnistNeuralNetworkKernels _networkKernels;

	public GpuBackend(Accelerator accelerator) {
		_accelerator = accelerator;

		Console.WriteLine($"Using Gpu Backend, on device: {accelerator.Name}");

		_arithmeticKernels = new ArithmeticKernels(accelerator);
		_matrixKernels = new MatrixKernels(accelerator);
		_reductionKernels = new ReductionKernels(accelerator);
		_functionKernels = new FunctionKernels(accelerator);
		_networkKernels = new MnistNeuralNetworkKernels(accelerator);
	}

	public ITensor CreateTensor(params int[] shape) {
		return new GpuTensor(_accelerator, shape);
	}

	public ITensor CreateTensor(float[] data, params int[] shape) {
		return new GpuTensor(_accelerator, data, shape);
	}

	public ISingleValue CreateValue() {
		return new GpuValue(_accelerator);
	}

	public ISingleValue CreateValue(float data) {
		return new GpuValue(_accelerator, data);
	}

	public ITensor Zero(params int[] shape) {
		return GpuTensor.Zero(_accelerator, shape);
	}

	public ITensor Random(Random random, float min, float max, params int[] shape) {
		return GpuTensor.Random(_accelerator, random, min, max, shape);
	}

	public void Set(ITensor a, float[] data) {
		((GpuTensor)a).Buffer.CopyFromCPU(data);
	}

	public float[] Read(ITensor a) {
		float[] r = new float[a.Length];
		_accelerator.Synchronize();
		((GpuTensor)a).Buffer.CopyToCPU(r);
		return r;
	}

	public void SetSingle(ISingleValue a, float data) {
		((GpuValue)a).Buffer.CopyFromCPU([data]);
	}

	public float ReadSingle(ISingleValue a) {
		float[] r = new float[1];
		_accelerator.Synchronize();
		((GpuValue)a).View.CopyToCPU(r);
		return r[0];
	}

	public void AddSingle(ISingleValue value, ISingleValue change) {
		GpuValue gValue = (GpuValue)value;
		GpuValue gChange = (GpuValue)change;

		_arithmeticKernels.Addition(1, gValue.View, gChange.View, gValue.View);
	}
	
	public void CheckAnswer(ISingleValue prediction, ISingleValue label, ISingleValue accuracyOut) {
		GpuValue gPrediction = (GpuValue)prediction;
		GpuValue gLabel = (GpuValue)label;
		GpuValue gAccuracyOut = (GpuValue)accuracyOut;
		
		_arithmeticKernels.IncreaseIfEqual(
			1,
			gPrediction.View,
			gLabel.View,
			gAccuracyOut.View
		);
	}

	public void MatrixTranspose(ITensor a, ITensor output) {
		GpuTensor gInput = (GpuTensor)a;
		GpuTensor gOutput = (GpuTensor)output;

		int outputSize = gInput.Length;

		int rows = gInput.Shape[0];
		int cols = gInput.Shape[1];

		_matrixKernels.Transpose(
			outputSize,
			rows,
			cols,
			gInput.View,
			gOutput.View
		);
	}

	public void Loss(ITensor activation, int label, ISingleValue lossOut) {
		GpuTensor gActivation = (GpuTensor)activation;
		GpuValue gLossOut = (GpuValue)lossOut;
		
		_arithmeticKernels.LogScalar(1, gActivation.View.SubView(label, 1), -1, gLossOut.View);
	}

	public void Input(ITensor input, ITensor activationOut) {
		GpuTensor gInput = (GpuTensor)input;
		GpuTensor gActivationOut = (GpuTensor)activationOut;

		int outputSize = activationOut.Shape[0];

		_networkKernels.Input(
			outputSize,
			gInput.View,
			gActivationOut.View
		);
	}

	public void ForwardPass(ITensor input, ITensor weights, ITensor bias, ITensor valuesOut) {
		GpuTensor gInput = (GpuTensor)input;
		GpuTensor gWeights = (GpuTensor)weights;
		GpuTensor gBias = (GpuTensor)bias;
		GpuTensor gValuesOut = (GpuTensor)valuesOut;

		int outputSize = valuesOut.Shape[0];
		int inputSize = input.Shape[0];

		_matrixKernels.Linear(
			outputSize,
			inputSize,
			gWeights.View,
			gInput.View,
			gBias.View,
			gValuesOut.View
		);
	}

	public void Activation(ITensor values, ITensor activationOut) {
		GpuTensor gValues = (GpuTensor)values;
		GpuTensor gActivationOut = (GpuTensor)activationOut;

		int outputSize = activationOut.Shape[0];

		_functionKernels.ReLu(
			outputSize,
			gValues.View,
			gActivationOut.View
		);
	}

	public void Output(ITensor values, ITensor activationOut, ISingleValue predictionOut) {
		GpuTensor gValues = (GpuTensor)values;
		GpuTensor gActivationOut = (GpuTensor)activationOut;
		GpuValue gPredictionOut = (GpuValue)predictionOut;

		int outputSize = activationOut.Shape[0];

		// max, so we avoid large values, as well as the prediction the network made
		GpuValue max = (GpuValue)CreateValue();
		_reductionKernels.MaxIndex(gValues, max, gPredictionOut);
			
		// e^(z[i] - max) for all i (i = row)
		GpuTensor e = (GpuTensor)Zero(gValues.Shape.ToArray());
		_arithmeticKernels.GpuConstantExponentiateSubtract(outputSize, gValues.View, max.View, e.View);

		// sum of e^z[i] for all i
		GpuValue sum = (GpuValue)CreateValue();
		_reductionKernels.Sum(e, sum);
		
		// e^z[i] / sum, aka the soft max
		_arithmeticKernels.GpuConstDivision(outputSize, e.View, sum.View, gActivationOut.View);
	
		_accelerator.Synchronize();

		max.Dispose();
		e.Dispose();
		sum.Dispose();
	}

	public void Backward(ITensor values, ITensor delta, ITensor weightsT, ITensor deltaOut) {
		GpuTensor gValues = (GpuTensor)values;
		GpuTensor gDelta = (GpuTensor)delta;
		GpuTensor gWeightsT = (GpuTensor)weightsT;
		GpuTensor gDeltaOut = (GpuTensor)deltaOut;

		int outputSize = deltaOut.Shape[0];
		int deltaSize = delta.Shape[0];

		_networkKernels.Backward(
			outputSize,
			gValues.View,
			gDelta.View,
			gWeightsT.View,
			gDeltaOut.View,
			deltaSize
		);
	}

	public void BackwardOutput(ITensor activations, ITensor expectedValues, ITensor deltaOut) {
		GpuTensor gActivations = (GpuTensor)activations;
		GpuTensor gExpectedValues = (GpuTensor)expectedValues;
		GpuTensor gDeltaOut = (GpuTensor)deltaOut;

		int outputSize = deltaOut.Shape[0];

		_arithmeticKernels.Subtraction(
			outputSize,
			gActivations.View,
			gExpectedValues.View,
			gDeltaOut.View
		);
	}

	public void BackPropagation(ITensor delta, ITensor activations, ITensor weightDelta, ITensor biasDelta) {
		GpuTensor gDelta = (GpuTensor)delta;
		GpuTensor gActivations = (GpuTensor)activations;
		GpuTensor gWeightsDelta = (GpuTensor)weightDelta;
		GpuTensor gBiasDelta = (GpuTensor)biasDelta;

		int columns = weightDelta.Shape[1];

		int outputSize = weightDelta.Length;
		_networkKernels.WeightGradient(
			outputSize,
			gDelta.View,
			gActivations.View,
			gWeightsDelta.View,
			columns);

		outputSize = biasDelta.Length;
		_networkKernels.BiasGradient(
			outputSize,
			gDelta.View,
			gBiasDelta.View);
	}

	public void Adjust(ITensor weightDelta, ITensor biasDelta, float scale, ITensor weights, ITensor weightsT,
		ITensor bias) {
		GpuTensor gWeightDelta = (GpuTensor)weightDelta;
		GpuTensor gBiasDelta = (GpuTensor)biasDelta;

		GpuTensor gWeights = (GpuTensor)weights;
		GpuTensor gWeightsT = (GpuTensor)weightsT;
		GpuTensor gBias = (GpuTensor)bias;

		int outputSize = gWeights.Length;
		_arithmeticKernels.ScalarSubtraction(
			outputSize,
			gWeights.View,
			gWeightDelta.View,
			scale,
			gWeights.View
		);

		int rows = gWeights.Shape[0];
		int cols = gWeights.Shape[1];

		_matrixKernels.Transpose(
			outputSize,
			rows, cols,
			gWeights.View,
			gWeightsT.View
		);

		outputSize = gBias.Length;
		_arithmeticKernels.ScalarSubtraction(
			outputSize,
			gBias.View,
			gBiasDelta.View,
			scale,
			gBias.View
		);
	}

	public void Synchronize() {
		_accelerator.Synchronize();
	}
}
	
public class ArithmeticKernels(Accelerator accelerator) {
	public readonly Action<Index1D, FloatView, FloatView,
		FloatView> Addition = accelerator
		.LoadAutoGroupedStreamKernel<Index1D, FloatView, FloatView, FloatView>(AdditionKernel);
		
	public readonly Action<Index1D, FloatView, float, FloatView>
		ConstAddition = accelerator
			.LoadAutoGroupedStreamKernel<Index1D, FloatView, float, FloatView>(AdditionKernel);
		
	public readonly Action<Index1D, FloatView, FloatView, float, FloatView>
		ScalarAddition = accelerator
			.LoadAutoGroupedStreamKernel<Index1D, FloatView, FloatView,float, FloatView>(AdditionKernel);
		
	public readonly Action<Index1D, FloatView, FloatViewSingle, FloatView>
		GpuConstAddition = accelerator
			.LoadAutoGroupedStreamKernel<Index1D, FloatView, FloatViewSingle, FloatView>(
				GpuConstantAdditionKernel);

	public readonly Action<Index1D, FloatViewSingle, FloatViewSingle, FloatViewSingle> IncreaseIfEqual =
		accelerator.LoadAutoGroupedStreamKernel<Index1D, FloatViewSingle, FloatViewSingle, FloatViewSingle>(
			IncreaseIfEqualKernel);
		
	public readonly Action<Index1D, FloatView, FloatView,
		FloatView> Subtraction = accelerator
		.LoadAutoGroupedStreamKernel<Index1D, FloatView, FloatView, FloatView>(SubtractionKernel);
		
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
		
	public readonly Action<Index1D, FloatView, float, FloatView> LogScalar = accelerator
		.LoadAutoGroupedStreamKernel<Index1D, FloatView, float, FloatView>(LogScalarKernel);

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
		FloatViewSingle inputA,
		FloatViewSingle inputB,
		FloatViewSingle output
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

	private static void LogScalarKernel(
		Index1D i,
		FloatView input,
		float constant,
		FloatView output
	) {
		output[i] = constant * XMath.Log(input[i]);
	}
}
	
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
	 * A Vector "o", size Rows
	 * </param>
	 *
	 * <returns>
	 * o = A * x + b
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
	
public class ReductionKernels(Accelerator accelerator) {
	public readonly Action<Index1D, int, FloatView, FloatView> MaxReduction =
		accelerator.LoadAutoGroupedStreamKernel<Index1D, int, FloatView, FloatView>(MaxReductionKernel);
	public readonly Action<Index1D, int, FloatView, FloatView> MinReduction =
		accelerator.LoadAutoGroupedStreamKernel<Index1D, int, FloatView, FloatView>(MinReductionKernel);
	public readonly Action<Index1D, int, FloatView, FloatView> SumReduction =
		accelerator.LoadAutoGroupedStreamKernel<Index1D, int, FloatView, FloatView>(SumReductionKernel);

	public readonly Action<Index1D, int, FloatView, FloatView, FloatView, FloatView> MaxIndexReduction =
		accelerator.LoadAutoGroupedStreamKernel<Index1D, int, FloatView, FloatView, FloatView, FloatView>(
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
	private void ReductionToValueAndIndex(GpuTensor input, GpuValue outputA, GpuValue outputB,
		Action<Index1D, int, FloatView, FloatView, FloatView, FloatView> kernel
	) {
		int count = input.Length;
			
		MemoryBuffer1D<float, Stride1D.Dense> inputValuesBuffer = accelerator.Allocate1D<float>(count);
		MemoryBuffer1D<float, Stride1D.Dense> inputIndicesBuffer = accelerator.Allocate1D<float>(count);
		MemoryBuffer1D<float, Stride1D.Dense> outputValuesBuffer = accelerator.Allocate1D<float>((count + 1) / 2);
		MemoryBuffer1D<float, Stride1D.Dense> outputIndicesBuffer = accelerator.Allocate1D<float>((count + 1) / 2);
			
		input.View.CopyTo(inputValuesBuffer.View);

		float[] indices = new float[count];
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
			.CopyTo(outputA.View.SubView(0,1));
		inputIndicesBuffer.View.SubView(0, 1)
			.CopyTo(outputB.View.SubView(0,1));
			
		accelerator.Synchronize();
			
		inputValuesBuffer.Dispose();
		inputIndicesBuffer.Dispose();
		outputValuesBuffer.Dispose();
		outputIndicesBuffer.Dispose();
	}

	public void MaxIndex(GpuTensor input, GpuValue outputValue, GpuValue outputIndex) {
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
	
	private static void MaxIndexReductionKernel(Index1D i, int length, FloatView inputValues, FloatView inputIndices, FloatView outputValues, FloatView outputIndices) {
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

public class FunctionKernels(Accelerator accelerator) {
	public readonly Action<Index1D, FloatView, FloatView> ReLu = accelerator
		.LoadAutoGroupedStreamKernel<Index1D, FloatView, FloatView>(ReLuKernel);
	public readonly Action<Index1D, FloatView, FloatView> Sigmoid = accelerator
		.LoadAutoGroupedStreamKernel<Index1D, FloatView, FloatView>(SigmoidKernel);
		
	private static void ReLuKernel(Index1D i, FloatView input, FloatView output) {
		output[i] = MathF.Max(input[i], 0);
	}
	private static void ReLuDerivativeKernel(Index1D i, FloatView input, FloatView output) {
		output[i] = input[i] > 0 ? 1 : 0;
	}
		
	private static void SigmoidKernel(Index1D i, FloatView input, FloatView output) {
		output[i] = 1 / (XMath.Exp(-input[i]) + 1);
	}
	private static void SigmoidDerivativeKernel(Index1D i, FloatView input, FloatView output) {
		float e = XMath.Exp(input[i]);
		output[i] = e / ((e + 1) * (e + 1));
	}
}
	
public class MnistNeuralNetworkKernels(Accelerator accelerator) {
	public readonly Action<Index1D, FloatView, FloatView> Input = accelerator
		.LoadAutoGroupedStreamKernel<Index1D, FloatView, FloatView>(InputKernel);

	public readonly Action<Index1D, FloatView, FloatView, FloatView, FloatView, int> Backward = accelerator
		.LoadAutoGroupedStreamKernel<Index1D, FloatView, FloatView, FloatView, FloatView, int>(BackwardKernel);

	public readonly Action<Index1D, FloatView, FloatView, FloatView, int> WeightGradient = accelerator
		.LoadAutoGroupedStreamKernel<Index1D, FloatView, FloatView, FloatView, int>(WeightGradientKernel);
	public readonly Action<Index1D, FloatView, FloatView> BiasGradient = accelerator
		.LoadAutoGroupedStreamKernel<Index1D, FloatView, FloatView>(BiasGradientKernel);
		
	private static void InputKernel(Index1D i, FloatView input, FloatView activationOut) {
		const float mean = 0.1307f;
		const float std = 0.3081f;
			
		activationOut[i] = (input[i] - mean) / std;
	}
		
	private static void BackwardKernel(
		Index1D row, FloatView values, FloatView delta, FloatView weightsT, FloatView deltaOut, int deltaSize
	) {
		float sum = 0;
			
		int offset = row * deltaSize;

		for (int col = 0; col < deltaSize; col++) {
			sum += weightsT[offset + col] * delta[col];
		}
			
		deltaOut[row] = values[row] > 0 ? sum : 0;
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