using NeuralNetwork.Core.Data;
using NeuralNetwork.Core.NetworkFormats;
using NeuralNetwork.Core.Others;
using NeuralNetwork.Core.Trackers;

namespace NeuralNetwork.Core.Iterators;

public abstract class NetworkScanner<TTracker> where TTracker : ScannerTracker {
	private readonly int _batchSize;
	private DataBatch<float> _dataBatch;
	private int _recycles;

	private bool _isRunning;
	
	public TTracker Tracker { get; }

	protected readonly int BufferSize;
	protected int DataSize => _dataBatch.Size;

	protected NetworkScanner(int batchSize, int bufferSize, int breathTicks = 2500) {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(batchSize);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(bufferSize);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(breathTicks);

        _batchSize = batchSize;
        BufferSize = bufferSize;
        Tracker = Activator.CreateInstance(typeof(TTracker), breathTicks) as TTracker
                  ?? throw new InvalidOperationException($"Could not create tracker of type {typeof(TTracker).Name}.");
	}
	
	/**
	 * Initializes the data batch that will be used for the iteration.
	 */
	public void LoadBatch(DataBatch<float> dataBatch, int batchRecycles = 1) {
		if (_isRunning)
			throw new InvalidOperationException($"Cannot replace the data batch while the scanner is running.");
		ArgumentNullException.ThrowIfNull(dataBatch);
		
		if (dataBatch.Size == 0)
			throw new ArgumentException("Data batch cannot be empty.", nameof(dataBatch));
		if (batchRecycles < 1)
			throw new ArgumentOutOfRangeException(nameof(batchRecycles), $"Recycles must be greater than 0.");
		
		_dataBatch = dataBatch.Copy();
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
	protected async Task<TimeSpan> IterateNetwork(Network network, Random random = null) {
		ArgumentNullException.ThrowIfNull(network);
		if (_dataBatch == null)
			throw new NullReferenceException("Attempted to iterate a network with no loaded data batch!");
		if (_isRunning)
			throw new InvalidOperationException("This scanner is already running.");

		_isRunning = true;
		OnStart();
		Tracker.Begin();

		try {
			for (int iteration = 0; iteration < _recycles && _isRunning; iteration++) {
				if (random != null) _dataBatch.Shuffle(random);
				OnNewSample();
				Tracker.BeginIteration();

				bool completedIteration = true;
				for (int batchPointer = 0; batchPointer < _dataBatch.Size; batchPointer += _batchSize) {
					if (!_isRunning) {
						completedIteration = false;
						break;
					}

					await ProcessRequest(
						network,
						_dataBatch.GetSmallBatch(
							batchPointer,
							Math.Min(_batchSize, _dataBatch.Size - batchPointer)
						)
					);
				}

				if (completedIteration) OnEndSample();
			}
		} finally {
			_isRunning = false;
			try {
				OnFinish();
			} finally {
				Tracker.Finish();
			}
		}

		return Tracker.TimeSinceStart;
	}

	private async Task ProcessRequest(Network network, DataBatch<float> trainingData) {
		if (Tracker.IsPaused) {
			await Tracker.AwaitPause();
		}
		if (!Tracker.IsRunning || !_isRunning) return;

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
		Action<TLocal, DataBatch<float>> processLocal,
		Action<TLocal> finalize
	) {
		TLocal local = createLocal();
		bool finalized = false;

		try {
			for (int i = 0; i < dataBatch.Size; i += BufferSize) {
				DataBatch<float> buffer = dataBatch.GetSmallBatch(i, BufferSize);
				processLocal(local, buffer);
			}

			finalize(local);
			finalized = true;
		} finally {
			if (!finalized && local is IDisposable disposable) {
				disposable.Dispose();
			}
		}
	}
	
	protected void ExecuteParallel<TLocal>(
		DataBatch<float> dataBatch,
		int processors,
		Func<TLocal> createLocal,
		Action<TLocal, DataBatch<float>> processLocal,
		Action<TLocal, Lock> finalize
	) {
		Lock lockObj = new Lock();
		int loops = 1 + (dataBatch.Size - 1) / BufferSize;
		
		MyParallel.For(0, loops, processors,
			createLocal,
			(i, local) => {
				DataBatch<float> buffer = dataBatch.GetSmallBatch(i * BufferSize, BufferSize);
				processLocal(local, buffer);
			},
			local => finalize(local, lockObj)
		);
	}
	
	public void ForceStop() {
		_isRunning = false;
		Tracker.Finish();
	}
	public void TogglePause() {
		Tracker.TogglePause();
	}
	public void Step() {
		Tracker.Step();
	}

	protected virtual void OnBreath() { }
	
	protected virtual void OnNewSample() { }
	protected virtual void OnEndSample() { }
	
	protected virtual void OnStart() { }
	protected virtual void OnFinish() { }
}