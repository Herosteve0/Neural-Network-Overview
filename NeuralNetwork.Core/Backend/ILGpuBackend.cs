using System.Numerics;
using ILGPU.Runtime;
using NeuralNetwork.Core.Backend.Kernels;
using NeuralNetwork.Core.Data.SingleValues;
using NeuralNetwork.Core.Data.Tensors;

namespace NeuralNetwork.Core.Backend;

/**
 * Gpu powered backend using an ILGPU accelerator.
 */
public class IlgpuBackend : IBackend, IMatrixBackend, IElementwiseBackend, IRandomBackend, IActivationBackend,
	ILossBackend {

	private readonly Accelerator _accelerator;

	private readonly ElementwiseKernels _elementwiseKernels;
	private readonly MatrixKernels _matrixKernels;
	private readonly ReductionKernels _reductionKernels;
	private readonly ActivationKernels _activationKernels;
	private readonly LossKernels _lossKernels;

	private static string ReadMemory(long memory) {
		string[] suffixes = ["B", "KB", "MB", "GB", "TB"];
		int index = 0;

		decimal scaledMemory = memory;
		
		while (scaledMemory >= 1024 && index < suffixes.Length - 1) {
			scaledMemory /= 1024m;
			index++;
		}

		return $"{scaledMemory:0.#}{suffixes[index]}";
	}
	
	public IlgpuBackend(Accelerator accelerator) {
		_accelerator = accelerator;

		Console.WriteLine($"Using Gpu Backend, on device: {accelerator.Name}. ({ReadMemory(accelerator.MemorySize)} Memory)");

		_elementwiseKernels = new ElementwiseKernels(accelerator);
		_matrixKernels = new MatrixKernels(accelerator);
		_reductionKernels = new ReductionKernels(accelerator);
		_activationKernels = new ActivationKernels(accelerator);
		_lossKernels = new LossKernels(accelerator);
	}

	/* IBackend */

	public ITensor<T> CreateTensor<T>(T value, params int[] shape) where T : unmanaged {
		return new IlgpuTensor<T>(_accelerator, value, shape);
	}

	public ITensor<T> CreateTensor<T>(T[] data, params int[] shape) where T : unmanaged {
		return new IlgpuTensor<T>(_accelerator, data, shape);
	}

	public ISingleValue<T> CreateValue<T>(T data) where T : unmanaged {
		return new IlgpuValue<T>(_accelerator, data);
	}

	public void Copy<T>(ITensor<T> from, ITensor<T> to) where T : unmanaged {
		((IlgpuTensor<T>)from).View.CopyTo(((IlgpuTensor<T>)to).View);
	}

	public void Set<T>(ITensor<T> a, T[] data) where T : unmanaged {
		((IlgpuTensor<T>)a).Buffer.CopyFromCPU(data);
	}

	public T[] Read<T>(ITensor<T> a) where T : unmanaged {
		T[] r = new T[a.Length];
		_accelerator.Synchronize();
		((IlgpuTensor<T>)a).Buffer.CopyToCPU(r);
		return r;
	}

	public void Set<T>(ISingleValue<T> a, T data) where T : unmanaged {
		((IlgpuValue<T>)a).Buffer.CopyFromCPU([data]);
	}

	public T Read<T>(ISingleValue<T> a) where T : unmanaged {
		T[] r = new T[1];
		_accelerator.Synchronize();
		((IlgpuValue<T>)a).View.CopyToCPU(r);
		return r[0];
	}

	public void AddSingle<T>(ISingleValue<T> value, ISingleValue<T> change) where T : unmanaged, INumber<T> {
		if (typeof(T) == typeof(float)) {
			IlgpuValue<float> gValue = (IlgpuValue<float>)value;
			IlgpuValue<float> gChange = (IlgpuValue<float>)change;

			_elementwiseKernels.IncreaseFloat(1, gValue.View, gChange.View);
		} else if (typeof(T) == typeof(int)) {
			IlgpuValue<int> gValue = (IlgpuValue<int>)value;
			IlgpuValue<int> gChange = (IlgpuValue<int>)change;

			_elementwiseKernels.IncreaseInt(1, gValue.View, gChange.View);
		} else {
			throw new Exception("Gpu addition only supports type: float, int.");
		}
	}

	public void CheckAnswer(ISingleValue<int> prediction, ISingleValue<int> label, ISingleValue<int> accuracyOut) {
		IlgpuValue<int> gPrediction = (IlgpuValue<int>)prediction;
		IlgpuValue<int> gLabel = (IlgpuValue<int>)label;
		IlgpuValue<int> gAccuracyOut = (IlgpuValue<int>)accuracyOut;

		_lossKernels.CheckAnswer(
			1,
			gPrediction.View,
			gLabel.View,
			gAccuracyOut.View
		);
	}

	public void Input(ITensor<float> input, ITensor<float> activationOut) {
		IlgpuTensor<float> gInput = (IlgpuTensor<float>)input;
		IlgpuTensor<float> gActivationOut = (IlgpuTensor<float>)activationOut;

		int outputSize = activationOut.Shape[0];

		_elementwiseKernels.Input(
			outputSize,
			gInput.View,
			gActivationOut.View
		);
	}

	public void MaxIndex(ITensor<float> input, ISingleValue<int> indexOut) {
		IlgpuTensor<float> gInput = (IlgpuTensor<float>)input;
		IlgpuValue<int> gIndexOut = (IlgpuValue<int>)indexOut;

		_reductionKernels.MaxIndex(gInput, null, gIndexOut);
	}

	public void Synchronize() {
		_accelerator.Synchronize();
	}

	/* IMatrixBackend */

	public void Linear(ITensor<float> input, ITensor<float> weights, ITensor<float> bias, ITensor<float> output) {
		IlgpuTensor<float> gInput = (IlgpuTensor<float>)input;
		IlgpuTensor<float> gWeights = (IlgpuTensor<float>)weights;
		IlgpuTensor<float> gBias = (IlgpuTensor<float>)bias;
		IlgpuTensor<float> gOutput = (IlgpuTensor<float>)output;

		int outputSize = output.Shape[0];
		int inputSize = input.Shape[0];

		_matrixKernels.Linear(
			outputSize,
			inputSize,
			gWeights.View,
			gInput.View,
			gBias.View,
			gOutput.View
		);
	}

	public void Delta(ITensor<float> delta, ITensor<float> weights, ITensor<float> deltaOut) {
		IlgpuTensor<float> gDelta = (IlgpuTensor<float>)delta;
		IlgpuTensor<float> gWeights = (IlgpuTensor<float>)weights;
		IlgpuTensor<float> gDeltaOut = (IlgpuTensor<float>)deltaOut;
		
		int outputSize = delta.Shape[0];
		int inputSize = deltaOut.Shape[0];

		_matrixKernels.Delta(
			inputSize,
			gDelta.View,
			gWeights.View,
			gDeltaOut.View,
			inputSize,
			outputSize
		);
	}

	public void BackPropagation(
		ITensor<float> delta, ITensor<float> activations, ITensor<float> weightDelta, ITensor<float> biasDelta
	) {
		IlgpuTensor<float> gDelta = (IlgpuTensor<float>)delta;
		IlgpuTensor<float> gActivations = (IlgpuTensor<float>)activations;
		IlgpuTensor<float> gWeightsDelta = (IlgpuTensor<float>)weightDelta;
		IlgpuTensor<float> gBiasDelta = (IlgpuTensor<float>)biasDelta;

		int columns = weightDelta.Shape[1];

		int outputSize = weightDelta.Length;
		_matrixKernels.WeightGradient(
			outputSize,
			gDelta.View,
			gActivations.View,
			gWeightsDelta.View,
			columns);

		outputSize = biasDelta.Length;
		_matrixKernels.BiasGradient(
			outputSize,
			gDelta.View,
			gBiasDelta.View);
	}

	/* IElementwiseBackend */

	public void Adjust(
		ITensor<float> weightDelta, ITensor<float> biasDelta, float scale, ITensor<float> weights, ITensor<float> bias
	) {
		IlgpuTensor<float> gWeights = (IlgpuTensor<float>)weights;
		IlgpuTensor<float> gBias = (IlgpuTensor<float>)bias;
		IlgpuTensor<float> gWeightDelta = (IlgpuTensor<float>)weightDelta;
		IlgpuTensor<float> gBiasDelta = (IlgpuTensor<float>)biasDelta;

		int outputSize = gWeights.Length;
		_elementwiseKernels.AdjustParameter(
			outputSize,
			gWeights.View,
			gWeightDelta.View,
			scale
		);

		outputSize = gBias.Length;
		_elementwiseKernels.AdjustParameter(
			outputSize,
			gBias.View,
			gBiasDelta.View,
			scale
		);
	}

	public void Dropout(ITensor<float> input, ITensor<float> mask, ITensor<float> output) {
		IlgpuTensor<float> gInput = (IlgpuTensor<float>)input;
		IlgpuTensor<float> gMask = (IlgpuTensor<float>)mask;
		IlgpuTensor<float> gOutput = (IlgpuTensor<float>)output;

		int length = gInput.Length;

		_elementwiseKernels.Dropout(
			length,
			gInput.View,
			gMask.View,
			gOutput.View
		);
	}
	
	/* IRandomBackend */
	
	public void GenerateDropoutMask(Random random, float dropoutChance, ITensor<float> maskOut) {
		IlgpuTensor<float> gMaskOut = (IlgpuTensor<float>)maskOut;

		float inverted = 1 / (1 - dropoutChance);
		int length = maskOut.Length;
		
		float[] values = new float[length];
		for (int i = 0; i < length; i++) {
			values[i] = random.NextSingle() >= dropoutChance
				? inverted
				: 0f;
		}
		
		gMaskOut.Buffer.CopyFromCPU(values);
	}

	/* IActivationBackend */

	public void ReLu_Forward(ITensor<float> input, ITensor<float> output) {
		IlgpuTensor<float> gInput = (IlgpuTensor<float>)input;
		IlgpuTensor<float> gOutput = (IlgpuTensor<float>)output;

		int outputSize = output.Length;

		_activationKernels.ReLuForward(
			outputSize,
			gInput.View,
			gOutput.View
		);
	}

	public void ReLu_Backward(ITensor<float> input, ITensor<float> output) {
		IlgpuTensor<float> gInput = (IlgpuTensor<float>)input;
		IlgpuTensor<float> gOutput = (IlgpuTensor<float>)output;

		int outputSize = output.Length;

		_activationKernels.ReLuBackward(
			outputSize,
			gInput.View,
			gOutput.View
		);
	}

	public void LeakyReLu_Forward(ITensor<float> input, float a, ITensor<float> output) {
		IlgpuTensor<float> gInput = (IlgpuTensor<float>)input;
		IlgpuTensor<float> gOutput = (IlgpuTensor<float>)output;

		int outputSize = output.Length;

		_activationKernels.LeakyReLuForward(
			outputSize,
			gInput.View,
			a,
			gOutput.View
		);
	}

	public void LeakyReLu_Backward(ITensor<float> input, float a, ITensor<float> output) {
		IlgpuTensor<float> gInput = (IlgpuTensor<float>)input;
		IlgpuTensor<float> gOutput = (IlgpuTensor<float>)output;

		int outputSize = output.Length;

		_activationKernels.LeakyReLuBackward(
			outputSize,
			gInput.View,
			a,
			gOutput.View
		);
	}

	public void Sigmoid_Forward(ITensor<float> input, ITensor<float> output) {
		IlgpuTensor<float> gInput = (IlgpuTensor<float>)input;
		IlgpuTensor<float> gOutput = (IlgpuTensor<float>)output;

		int outputSize = output.Length;

		_activationKernels.SigmoidForward(
			outputSize,
			gInput.View,
			gOutput.View
		);
	}

	public void Sigmoid_Backward(ITensor<float> input, ITensor<float> output) {
		IlgpuTensor<float> gInput = (IlgpuTensor<float>)input;
		IlgpuTensor<float> gOutput = (IlgpuTensor<float>)output;

		int outputSize = output.Length;

		_activationKernels.SigmoidBackward(
			outputSize,
			gInput.View,
			gOutput.View
		);
	}

	public void Softmax_Forward(ITensor<float> input, ITensor<float> output, ISingleValue<int> predictionOut) {
		IlgpuTensor<float> gInput = (IlgpuTensor<float>)input;
		IlgpuTensor<float> gOutput = (IlgpuTensor<float>)output;
		IlgpuValue<int> gPredictionOut = (IlgpuValue<int>)predictionOut;

		int outputSize = output.Length;

		IlgpuValue<float> value = null;
		try {
			// max, so we avoid large values, as well as the prediction the network made
			value = (IlgpuValue<float>)CreateValue(0f);
			_reductionKernels.MaxIndex(gInput, value, gPredictionOut);

			// e^(z[i] - max) for all i, where i = row
			_activationKernels.SoftmaxExp(outputSize, gInput.View, value.View, gOutput.View);

			// Σ e^z[i] for all i
			_reductionKernels.Sum(gOutput, value);

			// e^z[i] / sum, aka the soft max
			_activationKernels.SoftmaxDivide(outputSize, gOutput.View, value.View);

			// Synchronize because value will be disposed.
			_accelerator.Synchronize();
		} finally {
			try {
				_accelerator.Synchronize();
			} catch {  /* Ignored */ }
			value?.Dispose();
		}
	}

	public void Softmax_Backward(ITensor<float> input, ITensor<float> output) {
		IlgpuTensor<float> gInput = (IlgpuTensor<float>)input;
		IlgpuTensor<float> gOutput = (IlgpuTensor<float>)output;

		int length = gInput.Length;

		_activationKernels.SoftMaxBackward(length, length, gInput.View, gOutput.View);
	}

	/* ILossBackend */

	public void MSE_Calculate(
		ITensor<float> activation, ISingleValue<int> label, float a, ISingleValue<float> lossOut
	) {
		IlgpuTensor<float> gActivation = (IlgpuTensor<float>)activation;
		IlgpuValue<int> gLabel = (IlgpuValue<int>)label;
		IlgpuValue<float> gLossOut = (IlgpuValue<float>)lossOut;

		_lossKernels.MseCalculate(
			1,
			gActivation.View,
			gLabel.View,
			a,
			gLossOut.View
		);
	}

	public void MSE_Backward(ITensor<float> input, ITensor<float> expectedInput, float a, ITensor<float> output) {
		IlgpuTensor<float> gInput = (IlgpuTensor<float>)input;
		IlgpuTensor<float> gExpectedInput = (IlgpuTensor<float>)expectedInput;
		IlgpuTensor<float> gOutput = (IlgpuTensor<float>)output;

		int outputSize = output.Length;

		_lossKernels.MseBackward(
			outputSize,
			gInput.View,
			gExpectedInput.View,
			a,
			gOutput.View
		);
	}

	public void CrossEntropy_Calculate(
		ITensor<float> activation, ISingleValue<int> label, ISingleValue<float> lossOut
	) {
		IlgpuTensor<float> gActivation = (IlgpuTensor<float>)activation;
		IlgpuValue<int> gLabel = (IlgpuValue<int>)label;
		IlgpuValue<float> gLossOut = (IlgpuValue<float>)lossOut;

		_lossKernels.CrossEntropyCalculate(1, gActivation.View, gLabel.View, gLossOut.View);
	}

	public void CrossEntropy_Backward(
		ITensor<float> activations, ITensor<float> expectedValues, ITensor<float> deltaOut
	) {
		IlgpuTensor<float> gActivations = (IlgpuTensor<float>)activations;
		IlgpuTensor<float> gExpectedValues = (IlgpuTensor<float>)expectedValues;
		IlgpuTensor<float> gDeltaOut = (IlgpuTensor<float>)deltaOut;

		int outputSize = deltaOut.Length;

		_lossKernels.CrossEntropyBackward(
			outputSize,
			gActivations.View,
			gExpectedValues.View,
			gDeltaOut.View
		);
	}
}