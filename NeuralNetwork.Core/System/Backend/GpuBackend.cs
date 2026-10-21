using ILGPU.Runtime;
using NeuralNetwork.Core.System.Data;
using NeuralNetwork.Core.System.Backend.Kernels;

namespace NeuralNetwork.Core.System.Backend;

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

	public IValueFloat CreateValueFloat() {
		return new GpuValue(_accelerator);
	}
	public IValueInt CreateValueInt() {
		return new GpuInt(_accelerator);
	}
	public IValueFloat CreateValue(float data) {
		return new GpuValue(_accelerator, data);
	}
	public IValueInt CreateValue(int data) {
		return new GpuInt(_accelerator, data);
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
	
	public void SetSingle(IValueFloat a, float data) {
		((GpuValue)a).Buffer.CopyFromCPU([data]);
	}
	public float ReadSingle(IValueFloat a) {
		float[] r = new float[1];
		_accelerator.Synchronize();
		((GpuValue)a).View.CopyToCPU(r);
		return r[0];
	}
	public void SetSingle(IValueInt a, int data) {
		((GpuInt)a).Buffer.CopyFromCPU([data]);
	}
	public int ReadSingle(IValueInt a) {
		int[] r = new int[1];
		_accelerator.Synchronize();
		((GpuInt)a).View.CopyToCPU(r);
		return r[0];
	}

	public void AddSingle(IValueFloat value, IValueFloat change) {
		GpuValue gValue = (GpuValue)value;
		GpuValue gChange = (GpuValue)change;

		_arithmeticKernels.Addition(1, gValue.View, gChange.View, gValue.View);
	}
	public void AddSingle(IValueInt value, IValueInt change) {
		GpuInt gValue = (GpuInt)value;
		GpuInt gChange = (GpuInt)change;

		_arithmeticKernels.AdditionInt(1, gValue.View, gChange.View, gValue.View);
	}
	
	public void CheckAnswer(IValueInt prediction, IValueInt label, IValueInt accuracyOut) {
		GpuInt gPrediction = (GpuInt)prediction;
		GpuInt gLabel = (GpuInt)label;
		GpuInt gAccuracyOut = (GpuInt)accuracyOut;
		
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

	public void Loss(ITensor activation, int label, IValueFloat lossOut) {
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

	public void Output(ITensor values, ITensor activationOut, IValueInt predictionOut) {
		GpuTensor gValues = (GpuTensor)values;
		GpuTensor gActivationOut = (GpuTensor)activationOut;
		GpuInt gPredictionOut = (GpuInt)predictionOut;

		int outputSize = activationOut.Shape[0];

		// max, so we avoid large values, as well as the prediction the network made
		GpuValue max = (GpuValue)CreateValueFloat();
		_reductionKernels.MaxIndex(gValues, max, gPredictionOut);
			
		// e^(z[i] - max) for all i (i = row)
		GpuTensor e = (GpuTensor)Zero(gValues.Shape.ToArray());
		_arithmeticKernels.GpuConstantExponentiateSubtract(outputSize, gValues.View, max.View, e.View);

		// sum of e^z[i] for all i
		GpuValue sum = (GpuValue)CreateValueFloat();
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