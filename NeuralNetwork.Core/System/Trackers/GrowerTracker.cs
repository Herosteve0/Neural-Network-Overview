namespace NeuralNetwork.Core.System.Trackers;

public class GrowerTracker(TrainerTracker[] trainers, TesterTracker[] testers) {
	public int TotalGenerations { get; private set; }
	public int CurrentGeneration { get; private set; }

	public float Accuracy { get; private set; }
	public float Loss { get; private set; }

	public int TotalExamples => trainers[0].TotalExamples + testers[0].TotalExamples;
	public int Progress => (int)(trainers.Average(tracker => tracker.Progress) + testers.Average(tracker => tracker.Progress));

	public void Update() {
		int currentProgress = trainers.Sum(trainer => trainer.CurrentProgress);

		if (currentProgress == 0f) {
			Accuracy = 0;
			Loss = 0;
			return;
		}
        
		Accuracy = trainers.Average(tracker => tracker.Accuracy);
		Loss = trainers.Average(tracker => tracker.Loss);
	}
    
	public void Begin(int generations) {
		TotalGenerations = generations;
		CurrentGeneration = 0;
	}

	public void BeginGeneration() {
		CurrentGeneration++;
	}

	public GrowerProgressSnapshot Snapshot() {
		Update();
		return new GrowerProgressSnapshot(
			CurrentGeneration, TotalGenerations,
			Progress, TotalExamples,
			Loss, Accuracy
		);
	}
}

public readonly record struct GrowerProgressSnapshot(
	int Generation,
	int TotalGenerations,
	int Progress,
	int TotalExamples,
	float Loss,
	float Accuracy
);