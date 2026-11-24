using NeuralNetwork.Core.Data;
using NeuralNetwork.Core.Data.SingleValues;
using NeuralNetwork.Core.NetworkFormats.Layers.State;
using NeuralNetwork.Core.NetworkFormats.Networks;

namespace NeuralNetwork.Core.NetworkFormats.ExecutableNetworks;

public class InferenceNetwork : VirtualNetwork {
	private readonly ISingleValue<int> _correct;
	private readonly ISingleValue<int>[] _predictions;
	private readonly ILayerForwardState[] _states;
	
	public InferenceNetwork(Network network, int bufferSize = 1) : base(network, bufferSize) {
		ArgumentNullException.ThrowIfNull(network);
		ArgumentOutOfRangeException.ThrowIfNegativeOrZero(bufferSize);
		
		_correct = Backend.CreateValue(0);
		
		_predictions = new ISingleValue<int>[BufferSize];
		for (int i = 0; i < BufferSize; i++) {
			_predictions[i] = Backend.CreateValue(0);
		}
		
		_states = new ILayerForwardState[Length];
		for (int i = 0; i < _states.Length; i++) {
			_states[i] = network[i].CreateForwardState();
		}
	}
	
	public DataGuess<float> RawGuess(int index) {
		return new DataGuess<float>(
			InputBuffer[index], 
			Backend.Read(_predictions[index])
		);
	}

	public bool[] IsCorrect =>
		Enumerable.Range(0, InputBuffer.Size).Select(i => 
			Backend.Read(_predictions[i]) == InputBuffer[i].Label
		).ToArray();
	
	private ISingleValue<int> CurrentPrediction => _predictions[BufferIndex];
	
	public override void CompleteForwardPass(Network network) {
		IterateForward(network);
		EvaluateAnswer();
	}

	private void IterateForward(Network network) {
		for (int i = 0; i < network.LayerAmount; i++) {
			network[i].Forward(
				i == 0
					? CurrentInput
					: _states[i - 1].Output,
				_states[i],
				i == network.LayerAmount - 1
					? CurrentPrediction
					: null,
				PassMode.Inference
			);
		}
	}

	public void AdjustCorrect(ISingleValue<int> correctOut) {
		Backend.AddSingle(correctOut, _correct);
	}

	private void EvaluateAnswer() {
		Backend.CheckAnswer(CurrentPrediction, CurrentTarget, _correct);
	}

	protected override void Dispose(bool disposing) {
		if (!disposing) {
			base.Dispose(false);
			return;
		}
		
		try {
			_correct.Dispose();
				
			foreach (ISingleValue<int> prediction in _predictions) {
				prediction.Dispose();
			}

			foreach (ILayerForwardState state in _states) {
				state.Dispose();
			}
		} finally {
			base.Dispose(true);
		}
	}
}