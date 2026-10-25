using System.Numerics;
using NeuralNetwork.Core.Data.SingleValues;
using NeuralNetwork.Core.Data.Tensors;

namespace NeuralNetwork.Core.Backend;

public interface IBackend {
	/**
	 * Creates an appropriate tensor.
	 */
	ITensor<T> CreateTensor<T>(params int[] shape) where T : unmanaged;
	/**
	 * Creates an appropriate tensor filled with the given value.
	 */
	ITensor<T> CreateTensor<T>(T value, params int[] shape) where T : unmanaged;
	/**
	 * Creates an appropriate tensor with the given data.
	 * Data is copied.
	 */
	ITensor<T> CreateTensor<T>(T[] data, params int[] shape) where T : unmanaged;
	
	/**
	 * Creates an appropriate variable with the given data.
	 */
	ISingleValue<T> CreateValue<T>(T data) where T : unmanaged;

	/**
	 * Sets the tensor values to a given array.
	 */
	void Set<T>(ITensor<T> a, T[] data) where T : unmanaged;
	/**
	 * Converts the tensor into a cpu array.
	 */
	T[] Read<T>(ITensor<T> a) where T : unmanaged;
	
	/**
	 * Sets the value to a given value.
	 */
	void SetSingle<T>(ISingleValue<T> a, T value) where T : unmanaged;
	/**
	 * Converts the value to a cpu variable.
	*/
	T ReadSingle<T>(ISingleValue<T> a) where T : unmanaged;

	/**
	 * Adds a value to the first value.
	 */
	void AddSingle<T>(ISingleValue<T> value, ISingleValue<T> change) where T : unmanaged, INumber<T>;

	/**
	 * Checks whether the prediction was correct and increases a given accuracy value if so.
	 */
	public void CheckAnswer(ISingleValue<int> prediction, ISingleValue<int> label, ISingleValue<int> accuracyOut);
	
	/**
	 * Returns a Transposed Matrix.
	 */
	void MatrixTranspose(ITensor<float> a, ITensor<float> output);

	/**
	 * Returns the Loss
	 */
	void Loss(ITensor<float> activation, ISingleValue<int> label, ISingleValue<float> lossOut);
	
	/**
	 * The calculations for Activations of the first layer.
	 */
	void Input(ITensor<float> input, ITensor<float> activationOut);

	/**
	 * The calculation for the Values of a Layer.
	 */
	void ForwardPass(ITensor<float> input, ITensor<float> weights, ITensor<float> bias, ITensor<float> valuesOut);
		
	/**
	 * The calculations for the Activations of a Layer.
	 */
	void Activation(ITensor<float> values, ITensor<float> activationOut);
	/**
	 * The calculations for the Output of the last layer.
	 */
	void Output(ITensor<float> values, ITensor<float> activationOut, ISingleValue<int> predictionOut);

	/**
	 * The calculations for the Delta value that will be applied to the Weights and Biases.
	 */
	void Backward(ITensor<float> values, ITensor<float> delta, ITensor<float> weightsT, ITensor<float> deltaOut);
	/**
	 * The calculations for the initial Delta value at the output Layer.
	 */
	void BackwardOutput(ITensor<float> activations, ITensor<float> expectedValues, ITensor<float> deltaOut);
	/**
	 * The calculations for the change that will be applied to the Weights and Biases.
	 */
	void BackPropagation(ITensor<float> delta, ITensor<float> activations, ITensor<float> weightDelta, ITensor<float> biasDelta);
		
	/**
	 * Adjusts the Weights and Biases in proportion with the calculated Delta.
	 */
	void Adjust(ITensor<float> weightDelta, ITensor<float> biasDelta, float scale, ITensor<float> weights, ITensor<float> weightsT, ITensor<float> bias);

	void Synchronize();
}
	
// CPU

public abstract class CpuBackend : IBackend {
	public ITensor<T> CreateTensor<T>(params int[] shape) where T : unmanaged {
		return new CpuTensor<T>(shape);
	}
	public ITensor<T> CreateTensor<T>(T value, params int[] shape) where T : unmanaged {
		return new CpuTensor<T>(value, shape);
	}
	public ITensor<T> CreateTensor<T>(T[] data, params int[] shape) where T : unmanaged {
		return new CpuTensor<T>(data, shape);
	}
	
	public ISingleValue<T> CreateValue<T>(T data) where T : unmanaged {
		return new CpuValue<T>(data);
	}
	
	public void Set<T>(ITensor<T> a, T[] data) where T : unmanaged {
		((CpuTensor<T>)a).Data = data;
	}
	public T[] Read<T>(ITensor<T> a) where T : unmanaged {
		return ((CpuTensor<T>)a).Data.ToArray();
	}
	public void SetSingle<T>(ISingleValue<T> a, T data) where T : unmanaged {
		((CpuValue<T>)a).Value = data;
	}
	public T ReadSingle<T>(ISingleValue<T> a) where T : unmanaged {
		return ((CpuValue<T>)a).Value;
	}

	public void AddSingle<T>(ISingleValue<T> value, ISingleValue<T> change) where T : unmanaged, INumber<T> {
		CpuValue<T> cpuValue = (CpuValue<T>)value;
		CpuValue<T> cpuChange = (CpuValue<T>)change;

		cpuValue.Value += cpuChange;
	}

	public void MatrixTranspose(ITensor<float> a, ITensor<float> output) {
		CpuMatrixTranspose((CpuTensor<float>)a, (CpuTensor<float>)output);
	}

	public void CheckAnswer(ISingleValue<int> prediction, ISingleValue<int> label, ISingleValue<int> accuracyOut) {
		if (((CpuValue<int>)prediction).Value == ((CpuValue<int>)label).Value) {
			((CpuValue<int>)accuracyOut).Value++;
		}
	}
	
	public void Loss(ITensor<float> activation, ISingleValue<int> label, ISingleValue<float> lossOut) {
		CpuLoss((CpuTensor<float>)activation, (CpuValue<int>)label, (CpuValue<float>)lossOut);
	}

	public void Input(ITensor<float> input, ITensor<float> activationOut) {
		CpuInput((CpuTensor<float>)input, (CpuTensor<float>)activationOut);
	}

	public void ForwardPass(ITensor<float> input, ITensor<float> weights, ITensor<float> bias, ITensor<float> valuesOut) {
		CpuForwardPass((CpuTensor<float>)input, (CpuTensor<float>)weights, (CpuTensor<float>)bias, (CpuTensor<float>)valuesOut);
	}
	public void Activation(ITensor<float> values, ITensor<float> activationOut) {
		CpuActivation((CpuTensor<float>)values, (CpuTensor<float>)activationOut);
	}
	public void Output(ITensor<float> values, ITensor<float> activationOut, ISingleValue<int> predictionOut) {
		CpuOutput((CpuTensor<float>)values, (CpuTensor<float>)activationOut, (CpuValue<int>)predictionOut);
	}

	public void Backward(ITensor<float> values, ITensor<float> delta, ITensor<float> weightsT, ITensor<float> deltaOut) {
		CpuBackward((CpuTensor<float>)values, (CpuTensor<float>)delta, (CpuTensor<float>)weightsT, (CpuTensor<float>)deltaOut);
	}

	public void BackwardOutput(ITensor<float> activations, ITensor<float> expectedValues, ITensor<float> deltaOut) {
		CpuBackwardOutput((CpuTensor<float>)activations, (CpuTensor<float>)expectedValues, (CpuTensor<float>)deltaOut);
	}

	public void BackPropagation(ITensor<float> delta, ITensor<float> activations, ITensor<float> weightDelta, ITensor<float> biasDelta) {
		CpuBackPropagation((CpuTensor<float>)delta, (CpuTensor<float>)activations, (CpuTensor<float>)weightDelta, (CpuTensor<float>)biasDelta);
	}

	public void Adjust(ITensor<float> weightDelta, ITensor<float> biasDelta, float scale, ITensor<float> weights, ITensor<float> weightsT, ITensor<float> bias) {
		CpuAdjust((CpuTensor<float>)weightDelta, (CpuTensor<float>)biasDelta, scale, (CpuTensor<float>)weights, (CpuTensor<float>)weightsT, (CpuTensor<float>)bias);
	}

	public void Synchronize() { }

	protected abstract void CpuMatrixTranspose(CpuTensor<float> a, CpuTensor<float> output);

	protected abstract void CpuLoss(CpuTensor<float> activation, CpuValue<int> label, CpuValue<float> lossOut);
	
	protected abstract void CpuInput(CpuTensor<float> input, CpuTensor<float> activationOut);
		
	protected abstract void CpuForwardPass(CpuTensor<float> input, CpuTensor<float> weights, CpuTensor<float> bias, CpuTensor<float> valuesOut);
	protected abstract void CpuActivation(CpuTensor<float> values, CpuTensor<float> activationOut);
	protected abstract void CpuOutput(CpuTensor<float> values, CpuTensor<float> activationOut, CpuValue<int> predictionOut);
		
	protected abstract void CpuBackward(CpuTensor<float> values, CpuTensor<float> delta, CpuTensor<float> weightsT, CpuTensor<float> deltaOut);
	protected abstract void CpuBackwardOutput(CpuTensor<float> activations, CpuTensor<float> expectedValues, CpuTensor<float> deltaOut);
	protected abstract void CpuBackPropagation(CpuTensor<float> delta, CpuTensor<float> activations, CpuTensor<float> weightDelta, CpuTensor<float> biasDelta);
		
	protected abstract void CpuAdjust(CpuTensor<float> weightDelta, CpuTensor<float> biasDelta, float scale, CpuTensor<float> weights, CpuTensor<float> weightsT, CpuTensor<float> bias);
}