namespace NeuralNetwork.Core.Trackers;

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
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(dataSize);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(iterations);

        Iterations = iterations;
		DataSize = dataSize;
		TotalExamples = checked(dataSize * iterations);
	}

	protected virtual void Reset() {
		CurrentIteration = 0;
		Progress = 0;
		_breathCooldown = 0;
	}
	
	public void Begin() {
		Reset();
		
		_cancellationTokenSource?.Cancel();
		_cancellationTokenSource?.Dispose();
		_cancellationTokenSource = null;
		
		_isStepping = false;
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
	
	public void UpdateProgress() {
		BreathFunction();
	}


	public void TogglePause() {
		if (IsRunning) Pause();
		else if (IsRunning) Resume();
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
		
		CancellationTokenSource source = _cancellationTokenSource;
		_cancellationTokenSource = null;
		source?.Cancel();
		source?.Dispose();
	}

	public async Task AwaitPause() {
		CancellationTokenSource source = _cancellationTokenSource
			?? throw new InvalidOperationException("Paused tracker has no cancellation source.");

		try {
			await Task.Delay(Timeout.Infinite, source.Token);
		} catch (OperationCanceledException) {
			// Ignored
		} finally {
			if (ReferenceEquals(source, _cancellationTokenSource)) {
				_cancellationTokenSource = null;
				source?.Dispose();
			}
		}
	}

	public void CompleteStep() {
		if (!_isStepping) return;
		_isStepping = false;
		if (_state == ScannerState.Running) {
			Pause();
		}
	}

	public void Step() {
		if (!IsPaused) return;
		Resume(true);
	}
	
	public void Finish() {
		_state =  ScannerState.Idle;
		_isStepping = false;
		CancellationTokenSource source = _cancellationTokenSource;
		_cancellationTokenSource = null;
		source?.Cancel();
		source?.Dispose();
	}
}