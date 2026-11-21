using System.Numerics;
using NeuralNetwork.Core.Data.SingleValues;
using NeuralNetwork.Core.Data.Tensors;

namespace NeuralNetwork.Core.Backend;

public abstract class CpuBackend : IBackend, IMatrixBackend, IElementwiseBackend, IRandomBackend, IActivationBackend,
	ILossBackend {
	/* IBackend Implementation */
	public ITensor<T> CreateTensor<T>(T value, params int[] shape) where T : unmanaged {
		return new CpuTensor<T>(value, shape);
	}

	public ITensor<T> CreateTensor<T>(T[] data, params int[] shape) where T : unmanaged {
		return new CpuTensor<T>(data, shape);
	}

	public ISingleValue<T> CreateValue<T>(T data) where T : unmanaged {
		return new CpuValue<T>(data);
	}

	public void Copy<T>(ITensor<T> from, ITensor<T> to) where T : unmanaged {
		((CpuTensor<T>)from).Data.CopyTo(((CpuTensor<T>)to).Data);
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

	public void CheckAnswer(ISingleValue<int> prediction, ISingleValue<int> label, ISingleValue<int> accuracyOut) {
		if (((CpuValue<int>)prediction).Value == ((CpuValue<int>)label).Value) {
			((CpuValue<int>)accuracyOut).Value++;
		}
	}

	public void Input(ITensor<float> input, ITensor<float> activationOut) {
		CpuInput((CpuTensor<float>)input, (CpuTensor<float>)activationOut);
	}

	public void MaxIndex(ITensor<float> input, ISingleValue<int> indexOut) {
		CpuMaxIndex((CpuTensor<float>)input, (CpuValue<int>)indexOut);
	}

	public void Synchronize() {
	}

	/* IMatrixBackend Implementation */

	public void Linear(ITensor<float> input, ITensor<float> weights, ITensor<float> bias, ITensor<float> output) {
		CpuLinear((CpuTensor<float>)input, (CpuTensor<float>)weights, (CpuTensor<float>)bias, (CpuTensor<float>)output);
	}

	public void Delta(ITensor<float> delta, ITensor<float> weights, ITensor<float> deltaOut) {
		CpuDelta((CpuTensor<float>)delta, (CpuTensor<float>)weights, (CpuTensor<float>)deltaOut);
	}

	public void BackPropagation(
		ITensor<float> delta, ITensor<float> activations, ITensor<float> weightDelta,
		ITensor<float> biasDelta
	) {
		CpuBackPropagation((CpuTensor<float>)delta, (CpuTensor<float>)activations, (CpuTensor<float>)weightDelta,
			(CpuTensor<float>)biasDelta);
	}

	/* IElementwiseBackend */

	public void Adjust(
		ITensor<float> weightDelta, ITensor<float> biasDelta, float scale, ITensor<float> weights, ITensor<float> bias
	) {
		CpuAdjust((CpuTensor<float>)weightDelta, (CpuTensor<float>)biasDelta, scale, (CpuTensor<float>)weights,
			(CpuTensor<float>)bias);
	}

	public void Dropout(ITensor<float> input, ITensor<float> mask, ITensor<float> output) {
		CpuDropout((CpuTensor<float>)input, (CpuTensor<float>)mask, (CpuTensor<float>)output);
	}
	
	/* IRandomBackend */

	public void GenerateDropoutMask(Random random, float dropoutChance, ITensor<float> maskOut) {
		CpuTensor<float> cpuMaskOut = (CpuTensor<float>)maskOut;

		float inverted = 1f / (1 - dropoutChance);
		int length = maskOut.Length;
		
		for (int i = 0; i < length; i++) {
			cpuMaskOut[i] = random.NextSingle() > dropoutChance
				? inverted
				: 0f;
		}
	}

	/* IFunctionBackend */

	public void ReLu_Forward(ITensor<float> input, ITensor<float> output) {
		CpuReLuForward((CpuTensor<float>)input, (CpuTensor<float>)output);
	}

	public void ReLu_Backward(ITensor<float> input, ITensor<float> gradient) {
		CpuReLuBackward((CpuTensor<float>)input, (CpuTensor<float>)gradient);
	}

	public void LeakyReLu_Forward(ITensor<float> input, float a, ITensor<float> output) {
		CpuLeakyReLuForward((CpuTensor<float>)input, a, (CpuTensor<float>)output);
	}

	public void LeakyReLu_Backward(ITensor<float> input, float a, ITensor<float> gradient) {
		CpuLeakyReLuBackward((CpuTensor<float>)input, a, (CpuTensor<float>)gradient);
	}

	public void Sigmoid_Forward(ITensor<float> input, ITensor<float> output) {
		CpuSigmoidForward((CpuTensor<float>)input, (CpuTensor<float>)output);
	}

	public void Sigmoid_Backward(ITensor<float> input, ITensor<float> gradient) {
		CpuSigmoidBackward((CpuTensor<float>)input, (CpuTensor<float>)gradient);
	}

	public void Softmax_Forward(ITensor<float> input, ITensor<float> output, ISingleValue<int> predictionOut) {
		CpuSoftmaxForward((CpuTensor<float>)input, (CpuTensor<float>)output, (CpuValue<int>)predictionOut);
	}

	public void Softmax_Backward(ITensor<float> input, ITensor<float> gradient) {
		CpuSoftmaxBackward((CpuTensor<float>)input, (CpuTensor<float>)gradient);
	}

	/* ILossBackend */

	public void MSE_Calculate(
		ITensor<float> activation, ISingleValue<int> label, float a, ISingleValue<float> lossOut
	) {
		CpuMseCalculation((CpuTensor<float>)activation, (CpuValue<int>)label, a, (CpuValue<float>)lossOut);
	}

	public void MSE_Backward(ITensor<float> input, ITensor<float> expectedInput, float a, ITensor<float> output) {
		CpuMseBackward((CpuTensor<float>)input, (CpuTensor<float>)expectedInput, a, (CpuTensor<float>)output);
	}

	public void CrossEntropy_Calculate(
		ITensor<float> activation, ISingleValue<int> label, ISingleValue<float> lossOut
	) {
		CpuCrossEntropyCalculate((CpuTensor<float>)activation, (CpuValue<int>)label, (CpuValue<float>)lossOut);
	}

	public void CrossEntropy_Backward(ITensor<float> input, ITensor<float> expectedInput, ITensor<float> output) {
		CpuCrossEntropyBackward((CpuTensor<float>)input, (CpuTensor<float>)expectedInput, (CpuTensor<float>)output);
	}

	/* Non Cpu-Generic functions for the Scalar and Simd implementations */

	protected abstract void CpuInput(CpuTensor<float> input, CpuTensor<float> activationOut);

	protected abstract void CpuMaxIndex(CpuTensor<float> input, CpuValue<int> index);

	protected abstract void CpuLinear(
		CpuTensor<float> input, CpuTensor<float> weights, CpuTensor<float> bias,
		CpuTensor<float> valuesOut
	);

	protected abstract void CpuDelta(
		CpuTensor<float> delta, CpuTensor<float> weight,
		CpuTensor<float> deltaOut
	);

	protected abstract void CpuBackPropagation(
		CpuTensor<float> delta, CpuTensor<float> activations,
		CpuTensor<float> weightDelta, CpuTensor<float> biasDelta
	);

	protected abstract void CpuAdjust(
		CpuTensor<float> weightDelta, CpuTensor<float> biasDelta, float scale,
		CpuTensor<float> weights, CpuTensor<float> bias
	);

	protected abstract void CpuDropout(CpuTensor<float> input, CpuTensor<float> mask, CpuTensor<float> output);

	protected abstract void CpuReLuForward(CpuTensor<float> input, CpuTensor<float> output);
	protected abstract void CpuReLuBackward(CpuTensor<float> input, CpuTensor<float> gradient);

	protected abstract void CpuLeakyReLuForward(CpuTensor<float> input, float a, CpuTensor<float> output);
	protected abstract void CpuLeakyReLuBackward(CpuTensor<float> input, float a, CpuTensor<float> gradient);

	protected abstract void CpuSigmoidForward(CpuTensor<float> input, CpuTensor<float> output);
	protected abstract void CpuSigmoidBackward(CpuTensor<float> input, CpuTensor<float> gradient);

	protected abstract void CpuSoftmaxForward(
		CpuTensor<float> input, CpuTensor<float> output, CpuValue<int> predictionOut
	);

	protected abstract void CpuSoftmaxBackward(CpuTensor<float> input, CpuTensor<float> gradient);

	protected abstract void CpuMseCalculation(
		CpuTensor<float> activation, CpuValue<int> label, float a, CpuValue<float> lossOut
	);

	protected abstract void CpuMseBackward(
		CpuTensor<float> input, CpuTensor<float> expectedInput, float a, CpuTensor<float> output
	);

	protected abstract void CpuCrossEntropyCalculate(
		CpuTensor<float> activation, CpuValue<int> label, CpuValue<float> lossOut
	);

	protected abstract void CpuCrossEntropyBackward(
		CpuTensor<float> input, CpuTensor<float> expectedInput, CpuTensor<float> output
	);
}