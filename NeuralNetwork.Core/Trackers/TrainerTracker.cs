namespace NeuralNetwork.Core.Trackers;

public class TrainerTracker(int breathTicks) : ScannerTracker(breathTicks) {
	public float LearningRate { get; private set; }

	private float _loss;
	private int _correct;
	
	public float Loss => CurrentIteration == 0 ? 0 : _loss / CurrentIteration;
	public float Accuracy => Progress == 0 ? 0 : 100f * _correct / Progress;

	public override void Reset() {
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
