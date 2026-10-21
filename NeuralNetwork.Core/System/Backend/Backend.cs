using NeuralNetwork.Core.System.Data;

namespace NeuralNetwork.Core.System.Backend;

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
	IValueFloat CreateValueFloat();
	/**
	 * Creates an appropriate variable.
	 */
	IValueInt CreateValueInt();
	/**
	 * Creates an appropriate variable with the given data.
	 */
	IValueFloat CreateValue(float data);
	/**
	 * Creates an appropriate variable with the given data.
	 */
	IValueInt CreateValue(int data);
		
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
	void SetSingle(IValueFloat a, float data);
	/**
	 * Converts the value to a float.
	*/
	float ReadSingle(IValueFloat a);
	/**
	 * Sets the value to a given int.
	 */
	void SetSingle(IValueInt a, int data);
	/**
	 * Converts the value to an int.
	*/
	int ReadSingle(IValueInt a);

	/**
	 * Adds a value to the first value.
	 */
	void AddSingle(IValueFloat value, IValueFloat change);
	/**
	 * Adds a value to the first value.
	 */
	void AddSingle(IValueInt value, IValueInt change);

	/**
	 * Checks whether the prediction was correct and increases a given accuracy value if so.
	 */
	public void CheckAnswer(IValueInt prediction, IValueInt label, IValueInt accuracyOut);
	
	/**
	 * Returns a Transposed Matrix.
	 */
	void MatrixTranspose(ITensor a, ITensor output);

	/**
	 * Returns the Loss
	 */
	void Loss(ITensor activation, int label, IValueFloat lossOut);
	
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
	void Output(ITensor values, ITensor activationOut, IValueInt predictionOut);

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
	
	public IValueFloat CreateValueFloat() {
		return new CpuValue();
	}
	public IValueInt CreateValueInt() {
		return new CpuInt();
	}
	public IValueFloat CreateValue(float data) {
		return new CpuValue(data);
	}
	public IValueInt CreateValue(int data) {
		return new CpuInt(data);
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
	public void SetSingle(IValueFloat a, float data) {
		((CpuValue)a).Value = data;
	}
	public float ReadSingle(IValueFloat a) {
		return ((CpuValue)a).Value;
	}
	public void SetSingle(IValueInt a, int data) {
		((CpuInt)a).Value = data;
	}
	public int ReadSingle(IValueInt a) {
		return ((CpuInt)a).Value;
	}

	public void AddSingle(IValueFloat value, IValueFloat change) {
		((CpuValue)value).Value += (CpuValue)change;
	}
	public void AddSingle(IValueInt value, IValueInt change) {
		((CpuInt)value).Value += (CpuInt)change;
	}

	public void MatrixTranspose(ITensor a, ITensor output) {
		CpuMatrixTranspose((CpuTensor)a, (CpuTensor)output);
	}

	public void CheckAnswer(IValueInt prediction, IValueInt label, IValueInt accuracyOut) {
		if (((CpuInt)prediction).Value == ((CpuInt)label).Value) {
			((CpuInt)accuracyOut).Value++;
		}
	}
	
	public void Loss(ITensor activation, int label, IValueFloat lossOut) {
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
	public void Output(ITensor values, ITensor activationOut, IValueInt predictionOut) {
		CpuOutput((CpuTensor)values, (CpuTensor)activationOut, (CpuInt)predictionOut);
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
	protected abstract void CpuOutput(CpuTensor values, CpuTensor activationOut, CpuInt predictionOut);
		
	protected abstract void CpuBackward(CpuTensor values, CpuTensor delta, CpuTensor weightsT, CpuTensor deltaOut);
	protected abstract void CpuBackwardOutput(CpuTensor activations, CpuTensor expectedValues, CpuTensor deltaOut);
	protected abstract void CpuBackPropagation(CpuTensor delta, CpuTensor activations, CpuTensor weightDelta, CpuTensor biasDelta);
		
	protected abstract void CpuAdjust(CpuTensor weightDelta, CpuTensor biasDelta, float scale, CpuTensor weights, CpuTensor weightsT, CpuTensor bias);
}