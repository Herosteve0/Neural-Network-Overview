using NeuralNetwork.Core.Data;

namespace NeuralNetwork.Core;

public enum ScannerState {
	Idle,
	Running,
	Paused
}
public class ScannerTracker(int breathTicks) {
	public int TotalExamples { get; private set; }
	public int Iterations { get; private set; }
	protected int DataSize { get; private set; }

	private int CurrentProgress { get; set; }
	public int Progress { get; private set; }
	public float ProgressPercentage => (float)Progress / TotalExamples;
	private int _breathCooldown;

	private ScannerState _state = ScannerState.Idle;
	public bool IsRunning => _state != ScannerState.Idle;
	public bool IsPaused => _state is ScannerState.Paused;
	private bool _isStepping;
	private CancellationTokenSource _cancellationTokenSource;
	
	private DateTime _startTime;
	private DateTime _progressTime;
	public TimeSpan TimeSinceStart => DateTime.Now - _startTime;
	public TimeSpan TimeSinceLastProgress => DateTime.Now - _progressTime;

	public virtual void Initialize(int dataSize, int iterations) {
		Iterations = iterations;
		DataSize = dataSize;
		TotalExamples = dataSize * iterations;
	}

	public virtual void Reset() {
		Progress = 0;
	}
	
	public void Begin() {
		Reset();
		_startTime = DateTime.Now;
		_progressTime = DateTime.Now;
		
		_state = ScannerState.Running;
		
	}

	public virtual void BeginIteration() {
		CurrentProgress = 0;
	}
	
	public bool ApplyProgress(int progress) {
		CurrentProgress += progress;
		Progress += progress;
		_breathCooldown += progress;
		if (_breathCooldown >= breathTicks) {
			_breathCooldown = 0;
			return true;
		}
		return false;
	}

	protected virtual void BreathFunction() {
		_progressTime = DateTime.Now;
	}
	
	public TimeSpan UpdateProgress() {
		TimeSpan span = TimeSinceLastProgress;
		BreathFunction();
		return span;
	}

	public void Pause() {
		_state = ScannerState.Paused;
		_cancellationTokenSource = new CancellationTokenSource();
	}

	public void Resume() {
		Resume(false);
	}
	private void Resume(bool step) {
		_state = ScannerState.Running;
		_isStepping = step;
		_cancellationTokenSource.Cancel();
	}

	public async Task AwaitPause() {
		try {
			await Task.Delay(-1, _cancellationTokenSource.Token);
		} catch (Exception) {
			if (_isStepping) return;
			Pause();
		}
	}

	public void Step() {
		if (!IsPaused) return;
		Resume(true);
	}
	
	public TimeSpan Finish() {
		_state =  ScannerState.Idle;
		return TimeSinceStart;
	}
}

public abstract class NetworkScanner {
	protected int BatchSize { get; }
	private DataBatch<float> _dataBatch;
	private int _recycles;

	protected virtual ScannerTracker CreateTracker(int breathTicks) { 
		return new ScannerTracker(breathTicks);
	}

	protected ScannerTracker Tracker { get; }
	protected NetworkScanner(int batchSize, int breathTicks = 2500) {
		BatchSize = batchSize;
		// ReSharper disable once VirtualMemberCallInConstructor
		Tracker = CreateTracker(breathTicks);
	}

	/**
	 * Initializes the data batch that will be used for the iteration.
	 */
	public void LoadBatch(DataBatch<float> dataBatch, int batchRecycles = 1) {
		_dataBatch = dataBatch;
		_recycles = batchRecycles;
		
		Tracker.Initialize(dataBatch.Size, batchRecycles);
	}

	/**
	 * <param name="network">
	 * Network to iterate through.
	 * </param>
	 * <param name="random">
	 * If set, shuffles the Data Batch before each iteration, else it doesn't.
	 * </param>
	 *
	 * Call this function to process a network. Note: It expects a loaded Data Batch!
	 */
	protected async Task IterateNetwork(Network network, Random random = null) {
		if (_dataBatch == null) throw new NullReferenceException("Attempted to iterate a network with no loaded data batch!");
		if (_dataBatch.Size == 0) return;
		
		Tracker.Begin();
		for (int iteration = 0; iteration < _recycles; iteration++) {
			if (random != null) _dataBatch.Shuffle(random);
			OnNewSample();
			Tracker.BeginIteration();
			for (int batchPointer = 0; batchPointer < _dataBatch.Size; batchPointer += BatchSize) {
				if (!Tracker.IsRunning) {
					return;
				}
				
				await ProcessRequest(
					network,
					_dataBatch.GetSmallBatch(
						batchPointer,
						Math.Min(BatchSize, _dataBatch.Size - batchPointer)
					)
				);
			}
		}
	}

	private async Task ProcessRequest(Network network, DataBatch<float> trainingData) {
		if (Tracker.IsPaused) {
			await Tracker.AwaitPause();
		}

		Process(network, trainingData);

		bool breathe = Tracker.ApplyProgress(trainingData.Size);
		
		if (breathe) {
			Tracker.UpdateProgress();
			OnBreath();
			await Task.Delay(1);
		}
	}

	protected abstract void Process(Network network, DataBatch<float> dataBatch);

	protected void ExecuteSingle<TLocal>(
		DataBatch<float> dataBatch,
		Func<TLocal> createLocal,
		Action<TLocal, DataSample<float>> processLocal,
		Action<TLocal> finalize
	) {
		TLocal local = createLocal();

		foreach (DataSample<float> testingData in dataBatch.Data) {
			processLocal(local, testingData);
		}

		finalize(local);
	}
	
	protected void ExecuteParallel<TLocal>(
		DataBatch<float> dataBatch,
		Func<TLocal> createLocal,
		Action<TLocal, DataSample<float>> processLocal,
		Action<TLocal, Lock> finalize
	) {
		Lock lockObj = new Lock();
		
		Parallel.For(0, dataBatch.Size,
			createLocal,
			(i, _, local) => {
				DataSample<float> data = dataBatch[i];

				processLocal(local, data);
				
				return local;
			},
			(local) => finalize(local, lockObj)
		);
	}
	
	public void ForceStop() {
		Tracker.Finish();
	}
	public void TogglePause() {
		Tracker.Pause();
	}
	public void Step() {
		Tracker.Step();
	}

	protected virtual void OnBreath() { }
	protected virtual void OnNewSample() { }
}