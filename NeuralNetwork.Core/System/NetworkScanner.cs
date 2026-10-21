using NeuralNetwork.Core.System.Data;
using NeuralNetwork.Core.System.Trackers;

namespace NeuralNetwork.Core.System;

public abstract class NetworkScanner<TTracker>(int batchSize, int breathTicks = 2500) where TTracker : ScannerTracker {
	private DataBatch<float> _dataBatch;
	private int _recycles;

	private bool _isRunning;
	
	public TTracker Tracker { get; } = Activator.CreateInstance(typeof(TTracker), breathTicks) as TTracker;

	protected int DataSize => _dataBatch.Size;
	
	/**
	 * Initializes the data batch that will be used for the iteration.
	 */
	public void LoadBatch(DataBatch<float> dataBatch, int batchRecycles = 1) {
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
		if (_dataBatch == null) throw new NullReferenceException("Attempted to iterate a network with no loaded data batch!");
		if (_dataBatch.Size == 0) return TimeSpan.Zero;

		_isRunning = true;
		Tracker.Begin();
		for (int iteration = 0; iteration < _recycles; iteration++) {
			if (!_isRunning) break;
			
			if (random != null) _dataBatch.Shuffle(random);
			OnNewSample();
			Tracker.BeginIteration();
			for (int batchPointer = 0; batchPointer < _dataBatch.Size; batchPointer += batchSize) {
				if (!_isRunning) break;
				
				await ProcessRequest(
					network,
					_dataBatch.GetSmallBatch(
						batchPointer,
						Math.Min(batchSize, _dataBatch.Size - batchPointer)
					)
				);
			}
		}
		_isRunning = false;

		OnFinish();
		return Tracker.Finish();
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

		foreach (DataSample<float> testingData in dataBatch.DataSpan) {
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
		Tracker.TogglePause();
	}
	public void Step() {
		Tracker.Step();
	}

	protected virtual void OnBreath() { }
	protected virtual void OnNewSample() { }
	protected virtual void OnFinish() { }
}