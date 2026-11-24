using NeuralNetwork.Core.Backend;
using NeuralNetwork.Core.Data;
using NeuralNetwork.Core.Data.SingleValues;
using NeuralNetwork.Core.Data.Tensors;
using NeuralNetwork.Core.NetworkFormats.Layers;
using NeuralNetwork.Core.NetworkFormats.Layers.State;
using NeuralNetwork.Core.NetworkFormats.Networks;

namespace NeuralNetwork.Core.NetworkFormats.ExecutableNetworks;

public class TrainableNetwork : VirtualNetwork {
	private readonly ISingleValue<float> _sampleLoss;
	private readonly ISingleValue<float> _totalLoss;
	
	private readonly ISingleValue<int> _totalCorrect;

	private readonly ISingleValue<int>[] _predictions;
	private readonly ITensor<float>[] _expectedOutput;

	private readonly ILayerForwardState[] _forwardStates;
	private readonly ILayerBackwardState[] _backwardStates;

	private readonly LossFunction _lossFunction;

	public TrainableNetwork(Network network, int bufferSize = 1) : base(network, bufferSize) {
		ArgumentNullException.ThrowIfNull(network);
		ArgumentOutOfRangeException.ThrowIfNegativeOrZero(bufferSize);

		_lossFunction = network.LossFunction;

		_sampleLoss = Backend.CreateValue(0f);
		_totalLoss = Backend.CreateValue(0f);
		_totalCorrect = Backend.CreateValue(0);

		_predictions = new ISingleValue<int>[BufferSize];
		_expectedOutput = new ITensor<float>[BufferSize];
		for (int i = 0; i < BufferSize; i++) {
			_predictions[i] = Backend.CreateValue(0);
			_expectedOutput[i] = Backend.CreateTensor(0f, network.LayerLength[Length - 1]);
		}

		_forwardStates = new ILayerForwardState[Length];
		_backwardStates = new ILayerBackwardState[Length];
		for (int i = 0; i < Length; i++) {
			_forwardStates[i] = network[i].CreateForwardState();
			if (i == 0) continue;
			_backwardStates[i] = network[i].CreateBackwardState();
		}
	}

	protected override void FillBuffer(DataBatch<float> input) {
		base.FillBuffer(input);
		
		for (int i = 0; i < InputBuffer.Size; i++) {
			DataSample<float> sample = input[i];
			if ((uint)sample.Label >= (uint)_expectedOutput[i].Length)
				throw new ArgumentOutOfRangeException(nameof(input),
					$"Sample {i} has label {sample.Label}, but the output layer has {_expectedOutput[i].Length} neurons.");

			Backend.Set(_expectedOutput[i], sample.LabelArray(_expectedOutput[i].Length, 1f));
		}
	}

	public DataGuess<float> RawGuess(int index) {
		return new DataGuess<float>(InputBuffer[index], Backend.Read(_predictions[index]));
	}

	private ITensor<float> Result => _forwardStates[Length - 1].Output;
	private ISingleValue<int> CurrentPrediction => _predictions[BufferIndex];
	private ITensor<float> CurrentExpectedOutput => _expectedOutput[BufferIndex];

	public override void CompleteForwardPass(Network network) {
		IterateForward(network);
		EvaluateAnswer();
		EvaluateLoss();
	}

	private void IterateForward(Network network) {
		for (int i = 0; i < network.LayerAmount; i++) {
			network[i].Forward(i == 0 ? CurrentInput : _forwardStates[i - 1].Output, _forwardStates[i],
				i == network.LayerAmount - 1 ? CurrentPrediction : null, PassMode.Training);
		}
	}

	public void IterateBackward(Network network) {
		for (int i = network.LayerAmount - 1; i > 0; i--) {
			if (i == Length - 1) {
				_lossFunction.Backward(_forwardStates[i].Output, CurrentExpectedOutput, _backwardStates[i].Delta);
			} else {
				network[i + 1].Delta(_forwardStates[i + 1], _backwardStates[i + 1], _backwardStates[i].Delta);
			}

			if (network[i] is not ITrainableLayer layer) continue;

			if (i != Length - 1) {
				layer.ActivationDelta(_forwardStates[i], _backwardStates[i]);
			}

			layer.BackPropagation(_forwardStates[i - 1].Output, _backwardStates[i]);
		}
	}

	public void IterateAdjust(Network network, float scale) {
		for (int i = 1; i < network.LayerAmount; i++) {
			if (network[i] is not ITrainableLayer layer) continue;
			layer.Adjust(_backwardStates[i], scale);
		}
	}

	public void AdjustCorrect(ISingleValue<int> correctOut) {
		Backend.AddSingle(correctOut, _totalCorrect);
	}

	public void AdjustLoss(ISingleValue<float> lossOut) {
		Backend.AddSingle(lossOut, _totalLoss);
	}

	private void EvaluateAnswer() {
		Backend.CheckAnswer(CurrentPrediction, CurrentTarget, _totalCorrect);
	}

	private void EvaluateLoss() {
		_lossFunction.Calculate(Result, CurrentTarget, _sampleLoss);
		Backend.AddSingle(_totalLoss, _sampleLoss);
	}

	protected override void Dispose(bool disposing) {
		if (!disposing) {
			base.Dispose(false);
			return;
		}

		try {
			_sampleLoss.Dispose();
			_totalLoss.Dispose();
			_totalCorrect.Dispose();

			foreach (ISingleValue<int> prediction in _predictions) {
				prediction.Dispose();
			}
			foreach (ITensor<float> expectedOutput in _expectedOutput) {
				expectedOutput.Dispose();
			}

			foreach (ILayerForwardState state in _forwardStates) {
				state.Dispose();
			}
			foreach (ILayerBackwardState state in _backwardStates) {
				state?.Dispose();
			}
		} finally {
			base.Dispose(true);
		}
	}
}