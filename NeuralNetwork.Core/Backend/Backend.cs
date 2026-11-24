using System.Numerics;
using NeuralNetwork.Core.Data.SingleValues;
using NeuralNetwork.Core.Data.Tensors;

namespace NeuralNetwork.Core.Backend;

public interface IBackend {
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
	 * Copies the values of the first tensor to the second.
	 * Tensors must be of the same type.
	 */
	void Copy<T>(ITensor<T> from, ITensor<T> to) where T : unmanaged;
	
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
	void Set<T>(ISingleValue<T> a, T value) where T : unmanaged;
	/**
	 * Converts the value to a cpu variable.
	*/
	T Read<T>(ISingleValue<T> a) where T : unmanaged;

	/**
	 * Adds a value to the first value.
	 */
	void AddSingle<T>(ISingleValue<T> value, ISingleValue<T> change) where T : unmanaged, INumber<T>;

	/**
	 * Checks whether the prediction was correct and increases a given accuracy value if so.
	 */
	public void CheckAnswer(ISingleValue<int> prediction, ISingleValue<int> label, ISingleValue<int> accuracyOut);

	/**
	 * The calculations for Activations of the first layer.
	 */
	void Input(ITensor<float> input, ITensor<float> activationOut);

	void MaxIndex(ITensor<float> input, ISingleValue<int> indexOut);
	
	void Synchronize();
}

public interface IMatrixBackend {
	/**
	 * Sets the output to the input multiplied by the weights, plus the bias.
	 * <example>
	 * output = weights * input + bias
	 * </example>
	 */
	void Linear(ITensor<float> input, ITensor<float> weights, ITensor<float> bias, ITensor<float> output);
	
	/**
	 * The calculations for the Delta value that will be applied to the Weights and Biases.
	 */
	void Delta(ITensor<float> delta, ITensor<float> weights, ITensor<float> deltaOut);
	
	/**
	 * The calculations for the change that will be applied to the Weights and Biases.
	 */
	void BackPropagation(ITensor<float> delta, ITensor<float> activations, ITensor<float> weightDelta, ITensor<float> biasDelta);
}

public interface IElementwiseBackend {
	/**
	* Adjusts the Weights and Biases in proportion with the calculated Delta.
	*/
	void Adjust(ITensor<float> weightDelta, ITensor<float> biasDelta, float scale, ITensor<float> weights, ITensor<float> bias);
	
	void Dropout(ITensor<float> input, ITensor<float> mask, ITensor<float> output);
}

public interface IRandomBackend {
	void GenerateDropoutMask(Random random, float dropoutChance, ITensor<float> maskOut);
}

public interface IActivationBackend {
	void ReLu_Forward(ITensor<float> input, ITensor<float> output);
	void ReLu_Backward(ITensor<float> input, ITensor<float> output);

	void LeakyReLu_Forward(ITensor<float> input, float a, ITensor<float> output);
	void LeakyReLu_Backward(ITensor<float> input, float a, ITensor<float> output);

	void Sigmoid_Forward(ITensor<float> input, ITensor<float> output);
	void Sigmoid_Backward(ITensor<float> input, ITensor<float> output);
	
	void Softmax_Forward(ITensor<float> input, ITensor<float> output, ISingleValue<int> predictionOut);
	void Softmax_Backward(ITensor<float> input, ITensor<float> output);
}

public interface ILossBackend {
	void MSE_Calculate(ITensor<float> activation, ISingleValue<int> label, float a, ISingleValue<float> lossOut);
	void MSE_Backward(ITensor<float> input, ITensor<float> expectedInput, float a, ITensor<float> output);

	void CrossEntropy_Calculate(ITensor<float> activation, ISingleValue<int> label, ISingleValue<float> lossOut);
	void CrossEntropy_Backward(ITensor<float> input, ITensor<float> expectedInput,ITensor<float> output);
}