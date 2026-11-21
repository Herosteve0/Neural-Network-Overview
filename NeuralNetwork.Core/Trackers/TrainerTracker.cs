namespace NeuralNetwork.Core.Trackers;

public class TrainerTracker(int breathTicks) : ScannerTracker(breathTicks) {
	public float LearningRate { get; private set; }

	private float _loss;
	private int _correct;
	
	public float Loss => Progress == 0 ? 0 : _loss / Progress;
	public float Accuracy => Progress == 0 ? 0 : 100f * _correct / Progress;

	protected override void Reset() {
		base.Reset();
		
		_loss = 0f;
		_correct = 0;
	}

	public void Update(float learningRate, float loss, int correct) {
		LearningRate = learningRate;
		_loss += loss;
		_correct += correct;
	}
}
