namespace NeuralNetwork.Core.System.Trackers;

public enum ScannerState {
	Idle,
	Running,
	Paused
}

public abstract class ScannerTracker(int breathTicks) {
	public int TotalExamples { get; private set; }
	public int Iterations { get; private set; }
	public int CurrentIteration { get; private set; }
	protected int DataSize { get; private set; }

	public int CurrentProgress { get; private set; }
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
		CurrentIteration++;
		CurrentProgress = 0;
	}
	
	public bool ApplyProgress(int progress) {
		CurrentProgress += progress;
		Progress += progress;
		
		_breathCooldown += progress;
		if (_breathCooldown < breathTicks) return false;
		
		_breathCooldown = 0;
		return true;
	}

	protected virtual void BreathFunction() {
		_progressTime = DateTime.Now;
	}
	
	public TimeSpan UpdateProgress() {
		TimeSpan span = TimeSinceLastProgress;
		BreathFunction();
		return span;
	}


	public void TogglePause() {
		if (IsRunning) Pause();
		else Resume();
	}

	private void Pause() {
		if (IsPaused) return;
		_state = ScannerState.Paused;
		_cancellationTokenSource = new CancellationTokenSource();
	}

	private void Resume(bool step = false) {
		if (!IsPaused) return;
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
		Resume();
		return TimeSinceStart;
	}
}