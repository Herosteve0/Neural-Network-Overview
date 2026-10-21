using NeuralNetwork.Core.System.Data;

namespace NeuralNetwork.Core.System.Trackers;

public class TesterTracker(int breathTicks) : ScannerTracker(breathTicks) {
	private List<DataGuess<float>> _mistakes;

	public int Wrong => _mistakes.Count;
	public int Correct => TotalExamples - Wrong;
	public float Accuracy => Correct / (float)TotalExamples * 100;

	public override void Reset() {
		base.Reset();
		_mistakes = [];
	}

	public void Update(List<DataGuess<float>> mistakes) {
		_mistakes.AddRange(mistakes);
	}

	public DataGuess<float>[] GetMistakes() {
		return _mistakes.ToArray();
	}
}