using System.Numerics;
using ILGPU.Runtime;
using NeuralNetwork.Core.Backend.Kernels;
using NeuralNetwork.Core.Data.SingleValues;
using NeuralNetwork.Core.Data.Tensors;

namespace NeuralNetwork.Core.Backend;

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

	public ITensor<T> CreateTensor<T>(params int[] shape) where T : unmanaged {
		return new GpuTensor<T>(_accelerator, shape);
	}
	public ITensor<T> CreateTensor<T>(T value, params int[] shape) where T : unmanaged {
		return new GpuTensor<T>(_accelerator, value, shape);
	}
	public ITensor<T> CreateTensor<T>(T[] data, params int[] shape) where T : unmanaged {
		return new GpuTensor<T>(_accelerator, data, shape);
	}
	
	public ISingleValue<T> CreateValue<T>(T data) where T : unmanaged {
		return new GpuValue<T>(_accelerator, data);
	}
	
	public void Set<T>(ITensor<T> a, T[] data) where T : unmanaged {
		((GpuTensor<T>)a).Buffer.CopyFromCPU(data);
	}
	public T[] Read<T>(ITensor<T> a) where T : unmanaged {
		T[] r = new T[a.Length];
		_accelerator.Synchronize();
		((GpuTensor<T>)a).Buffer.CopyToCPU(r);
		return r;
	}
	
	public void SetSingle<T>(ISingleValue<T> a, T data) where T : unmanaged {
		((GpuValue<T>)a).Buffer.CopyFromCPU([data]);
	}
	public T ReadSingle<T>(ISingleValue<T> a) where T : unmanaged {
		T[] r = new T[1];
		_accelerator.Synchronize();
		((GpuValue<T>)a).View.CopyToCPU(r);
		return r[0];
	}

	public void AddSingle<T>(ISingleValue<T> value, ISingleValue<T> change) where T : unmanaged, INumber<T> {
		if (typeof(T) == typeof(float)) {
			GpuValue<float> gValue = (GpuValue<float>)value;
			GpuValue<float> gChange = (GpuValue<float>)change;
			
			_arithmeticKernels.Addition(1, gValue.View, gChange.View, gValue.View);
		} else if (typeof(T) == typeof(int)) {
			GpuValue<int> gValue = (GpuValue<int>)value;
			GpuValue<int> gChange = (GpuValue<int>)change;
			
			_arithmeticKernels.AdditionInt(1, gValue.View, gChange.View, gValue.View);
		} else {
			throw new Exception("Gpu addition only supports type: float, int.");
		}
	}
	
	public void CheckAnswer(ISingleValue<int> prediction, ISingleValue<int> label, ISingleValue<int> accuracyOut) {
		GpuValue<int> gPrediction = (GpuValue<int>)prediction;
		GpuValue<int> gLabel = (GpuValue<int>)label;
		GpuValue<int> gAccuracyOut = (GpuValue<int>)accuracyOut;
		
		_arithmeticKernels.IncreaseIfEqual(
			1,
			gPrediction.View,
			gLabel.View,
			gAccuracyOut.View
		);
	}

	public void MatrixTranspose(ITensor<float> a, ITensor<float> output) {
		GpuTensor<float> gInput = (GpuTensor<float>)a;
		GpuTensor<float> gOutput = (GpuTensor<float>)output;

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

	public void Loss(ITensor<float> activation, ISingleValue<int> label, ISingleValue<float> lossOut) {
		GpuTensor<float> gActivation = (GpuTensor<float>)activation;
		GpuValue<int> gLabel = (GpuValue<int>)label;
		GpuValue<float> gLossOut = (GpuValue<float>)lossOut;
		
		_networkKernels.Loss(1, gActivation.View, gLabel.View, gLossOut.View);
	}

	public void Input(ITensor<float> input, ITensor<float> activationOut) {
		GpuTensor<float> gInput = (GpuTensor<float>)input;
		GpuTensor<float> gActivationOut = (GpuTensor<float>)activationOut;

		int outputSize = activationOut.Shape[0];

		_networkKernels.Input(
			outputSize,
			gInput.View,
			gActivationOut.View
		);
	}

	public void ForwardPass(ITensor<float> input, ITensor<float> weights, ITensor<float> bias, ITensor<float> valuesOut) {
		GpuTensor<float> gInput = (GpuTensor<float>)input;
		GpuTensor<float> gWeights = (GpuTensor<float>)weights;
		GpuTensor<float> gBias = (GpuTensor<float>)bias;
		GpuTensor<float> gValuesOut = (GpuTensor<float>)valuesOut;

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

	public void Activation(ITensor<float> values, ITensor<float> activationOut) {
		GpuTensor<float> gValues = (GpuTensor<float>)values;
		GpuTensor<float> gActivationOut = (GpuTensor<float>)activationOut;

		int outputSize = activationOut.Shape[0];

		_functionKernels.ReLu(
			outputSize,
			gValues.View,
			gActivationOut.View
		);
	}

	public void Output(ITensor<float> values, ITensor<float> activationOut, ISingleValue<int> predictionOut) {
		GpuTensor<float> gValues = (GpuTensor<float>)values;
		GpuTensor<float> gActivationOut = (GpuTensor<float>)activationOut;
		GpuValue<int> gPredictionOut = (GpuValue<int>)predictionOut;

		int outputSize = activationOut.Shape[0];

		// max, so we avoid large values, as well as the prediction the network made
		GpuValue<float> value = (GpuValue<float>)CreateValue(0f);
		_reductionKernels.MaxIndex(gValues, value, gPredictionOut);
			
		// e^(z[i] - max) for all i (i = row)
		GpuTensor<float> e = (GpuTensor<float>)CreateTensor<float>(gValues.Shape.ToArray());
		_arithmeticKernels.GpuConstantExponentiateSubtract(outputSize, gValues.View, value.View, e.View);

		// sum of e^z[i] for all i
		_reductionKernels.Sum(e, value);
		
		// e^z[i] / sum, aka the soft max
		_arithmeticKernels.GpuConstDivision(outputSize, e.View, value.View, gActivationOut.View);
	
		_accelerator.Synchronize();

		e.Dispose();
		value.Dispose();
	}

	public void Backward(ITensor<float> values, ITensor<float> delta, ITensor<float> weightsT, ITensor<float> deltaOut) {
		GpuTensor<float> gValues = (GpuTensor<float>)values;
		GpuTensor<float> gDelta = (GpuTensor<float>)delta;
		GpuTensor<float> gWeightsT = (GpuTensor<float>)weightsT;
		GpuTensor<float> gDeltaOut = (GpuTensor<float>)deltaOut;

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

	public void BackwardOutput(ITensor<float> activations, ITensor<float> expectedValues, ITensor<float> deltaOut) {
		GpuTensor<float> gActivations = (GpuTensor<float>)activations;
		GpuTensor<float> gExpectedValues = (GpuTensor<float>)expectedValues;
		GpuTensor<float> gDeltaOut = (GpuTensor<float>)deltaOut;

		int outputSize = deltaOut.Shape[0];

		_arithmeticKernels.Subtraction(
			outputSize,
			gActivations.View,
			gExpectedValues.View,
			gDeltaOut.View
		);
	}

	public void BackPropagation(ITensor<float> delta, ITensor<float> activations, ITensor<float> weightDelta, ITensor<float> biasDelta) {
		GpuTensor<float> gDelta = (GpuTensor<float>)delta;
		GpuTensor<float> gActivations = (GpuTensor<float>)activations;
		GpuTensor<float> gWeightsDelta = (GpuTensor<float>)weightDelta;
		GpuTensor<float> gBiasDelta = (GpuTensor<float>)biasDelta;

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

	public void Adjust(ITensor<float> weightDelta, ITensor<float> biasDelta, float scale, ITensor<float> weights, ITensor<float> weightsT, ITensor<float> bias) {
		GpuTensor<float> gWeightDelta = (GpuTensor<float>)weightDelta;
		GpuTensor<float> gBiasDelta = (GpuTensor<float>)biasDelta;

		GpuTensor<float> gWeights = (GpuTensor<float>)weights;
		GpuTensor<float> gWeightsT = (GpuTensor<float>)weightsT;
		GpuTensor<float> gBias = (GpuTensor<float>)bias;

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