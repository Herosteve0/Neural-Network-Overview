using Neurtal_Network_Overview.System.Data;

namespace Neurtal_Network_Overview.System;

public enum ScannerState {
	Idle,
	Active,
	Paused,
	Step
}
public class ScannerInfo {
	private ScannerState _state = ScannerState.Idle;
	public int MaxSteps { get; private set; } = 0;
	public int Steps { get; private set; } = 0;
	private DateTime _timeTracker;
	private CancellationTokenSource _canceltoken;
        
	public bool IsRunning =>  _state == ScannerState.Active;
	public bool IsActive =>  _state != ScannerState.Idle;
	public bool IsPaused =>  _state == ScannerState.Paused;
	public bool IsStep =>  _state == ScannerState.Step;
        
	public bool ShouldPause =>  _state is ScannerState.Paused or ScannerState.Step;
	public bool TrainingComplete => Steps == MaxSteps && MaxSteps != 0;
        
	public void Begin(int maxSteps) {
		_state = ScannerState.Active;
		MaxSteps = maxSteps;
		Steps = 0;
		_timeTracker = DateTime.Now;
	}
        
	public void AddProgress(int step) {
		Steps += step;
	}

	public async Task TestPause() {
		if (!ShouldPause) return;
	        
		try {
			await Task.Delay(-1, _canceltoken.Token);
		}
		catch (TaskCanceledException) {
			if (IsStep) _canceltoken = new CancellationTokenSource();
		}
	}

	public void TogglePause() {
		if (!IsActive) return;

		if (_state != ScannerState.Paused) {
			_state = ScannerState.Paused;
			_canceltoken = new CancellationTokenSource();
		} else {
			_state = ScannerState.Active;
			_canceltoken.Cancel();
		}
	}
	public void ToggleStep() {
		if (!IsActive) return;
            
		if (_state != ScannerState.Step) {
			_state = ScannerState.Step;
			_canceltoken = new CancellationTokenSource();
		} else {
			_state = ScannerState.Active;
			_canceltoken.Cancel();
		}
	}

	public void Step() {
		if (!IsStep) return;
	        
		_canceltoken.Cancel();
	}
        
	public TimeSpan Completed() {
		_state = ScannerState.Idle;
		return DateTime.Now - _timeTracker;
	}
}
	
public abstract class NetworkScanner {
	public NetworkScanner(string id, NeuralNetwork network, int batchSize, int delayTicks = 2500, bool isSilent = false) {
		Id = id;
		Network = network;
		BatchSize = batchSize;
			
		_delayTicks = delayTicks;
		_info = new ScannerInfo();
		IsSilent = isSilent;
	}

	private string Id { get; }
	protected NeuralNetwork Network { get; }
	protected IBackend Backend => Network.Backend;

	private int BatchSize { get; }
	private readonly ScannerInfo _info;
	protected readonly bool IsSilent;
	private readonly int _delayTicks;
	private DateTime _deltaTime;

	protected async Task Scan(DataBatch<float> scanData, int repeats = 1, Random random = null) {
		int totalExamples = scanData.Size * repeats;
		_info.Begin(totalExamples);
		_deltaTime = DateTime.Now;
		PrintMessage(ConsoleMessages.Start, totalExamples);

		int counter = 0;
		for (int cycle = 0; cycle < repeats; cycle++) {
			if (random != null) scanData.Shuffle(random);
			for (int i = 0; i < scanData.Size; i += BatchSize) {
				if (!_info.IsActive) return;
				counter += BatchSize;

				bool breathe = counter >= _delayTicks;
				await ProcessRequest(scanData.GetSmallBatch(i, Math.Min(BatchSize, scanData.Size - i)), breathe);
				if (breathe) counter = 0;
			}
		}

		PrintMessage(ConsoleMessages.Finish, _info.Completed());
	}

	private async Task ProcessRequest(DataBatch<float> trainingData, bool breathe) {
		await _info.TestPause();

		Process(trainingData);
		_info.AddProgress(trainingData.Size);

		if (breathe) {
			PrintMessage(ConsoleMessages.Progress, _info.Steps, _info.MaxSteps, _deltaTime);
			BreathFunction();
			_deltaTime = DateTime.Now;
			await Task.Delay(1);
		}
	}

	protected abstract void Process(DataBatch<float> dataBatch);

        
	public void ForceStop(string message) {
		if (!_info.IsActive) return;
		_info.Completed();
		PrintMessage(ConsoleMessages.ForceStop, message);
	}
	public void TogglePause() {
		_info.TogglePause();
		PrintMessage(ConsoleMessages.Pause, _info.IsPaused);
	}
	public void ToggleStep() {
		_info.ToggleStep();
		PrintMessage(ConsoleMessages.Step, _info.IsStep);
	}
	public void DoStep() {
		if (!_info.IsStep) return;
		_info.Step();
		PrintMessage(ConsoleMessages.DoStep);
	}
        
	protected enum ConsoleMessages {
		Start,
		Progress,
		Finish,
		ForceStop,
		Pause,
		Step,
		DoStep,

		InUse
	}
	protected void PrintMessage(ConsoleMessages type, params object[] args) {
		if (IsSilent) return;
		Console.WriteLine($"[{Id}]: " + type switch {
			ConsoleMessages.Start => $"Started process on a {args[0]} Data Batch.",
			ConsoleMessages.Progress =>
				$"Process is {100d * (int)args[0] / (int)args[1]:F2}% Complete [{args[0]}/{args[1]}], Estimated time left: {(_info.MaxSteps - _info.Steps) * ((DateTime.Now - _deltaTime) / _delayTicks)}",
			ConsoleMessages.Finish => $"Process Complete. [{(TimeSpan)args[0]}]",
			ConsoleMessages.ForceStop => $"Force stopped process: {args[0]}",
			ConsoleMessages.Pause => $"{((bool)args[0] ? "Paused" : "Unpaused")} + process.",
			ConsoleMessages.Step => $"{((bool)args[0] ? "Paused" : "Unpaused")} + process.",
			ConsoleMessages.DoStep => $"Did one process step.",
		        
			ConsoleMessages.InUse => $"Wait until all other processes are finished before beginning a new one.",

			_ => throw new ArgumentOutOfRangeException(nameof(type), type, null)
		});
	}

	protected virtual void BreathFunction() { }
}