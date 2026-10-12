using ILGPU.Runtime;
using Neurtal_Network_Overview.System.Data;

namespace Neurtal_Network_Overview.System;

public class GpuMultiTrainer(
	int trainerAmount,
	Accelerator accelerator,
	int[] layers,
	DataBatch<float> trainingData,
	DataBatch<float> testingData) {
		
	private readonly IBackend _backend = new GpuBackend(accelerator);
	private readonly int _workers = Math.Clamp(trainerAmount, 1, Environment.ProcessorCount);

	private DateTime _startTime;
	private DateTime _deltaTime;

	public async Task<NetworkInstance> TrainProcess(int randomSeed, int repeats, int iterations) {
		Random random = new Random(randomSeed);
		NetworkInstance network = await CreateNetwork(random);
		float learningRate = 0.025f;
		float mutationStrength = 0.025f;
			
		Console.WriteLine($"Training {_workers} networks over {iterations} iterations using {trainingData.Size * repeats} examples.");
			
		NetworkEvolver evolver = new NetworkEvolver(random, trainingData, testingData);
			
		_startTime = DateTime.Now;
		for (int cycle = 0; cycle < iterations; cycle++) {
			_deltaTime = DateTime.Now;
			(network, learningRate) = await evolver.GrowMutations(_workers, learningRate, mutationStrength, network, repeats, randomSeed);
			mutationStrength *= 0.75f;
			Console.WriteLine($"[{cycle + 1}/{iterations}] Estimated time left: {(iterations - cycle - 1) * (DateTime.Now - _deltaTime)}");
		}
			
		Console.WriteLine($"Multi Training complete in {DateTime.Now - _startTime}");
			
		return network;
	}

	private async Task<NetworkInstance> CreateNetwork(Random random) {
		NeuralNetwork network = new NeuralNetwork(layers, _backend, random);
		Tester tester = NetworkGrower.CreateTester(network);
		int accuracy = await tester.Test(testingData);
		return new NetworkInstance(network, accuracy);
	}
}

internal class NetworkEvolver(
	Random random,
	DataBatch<float> trainingData,
	DataBatch<float> testingData,
	bool isSilent = false) : NetworkGrower(trainingData, testingData, isSilent) {
	private readonly bool _isSilent = isSilent;

	public async Task<(NetworkInstance, float)> GrowMutations(int workers, float learningRate, float mutationStrength, NetworkInstance referenceNetwork, int repeats, int randomSeed) {
		DateTime deltaTime = DateTime.Now;
		if (!_isSilent) Console.WriteLine($"[Network Evolver] Begun mutation over {workers} networks.");
			
		(NetworkInstance[] newNetworks, float newLearningRate) = await TrainInParallel(workers, learningRate, mutationStrength, referenceNetwork.Network, repeats, randomSeed);
		NetworkInstance bestNetwork = FindBestNetworkAndDiscardRest(referenceNetwork, newNetworks);

		if (!_isSilent) {
			Console.WriteLine($"[Network Grower] Best Accuracy over {workers} workers: {bestNetwork.Accuracy} | " +
			                  $"Average Accuracy: {learningRate} | " + $"Time elapsed: {DateTime.Now - deltaTime}");
			Console.WriteLine();
		}
			

		return (bestNetwork, newLearningRate);
	}
		
	private async Task<(NetworkInstance[], float)> TrainInParallel(int workers, float learningRate, float mutationStrength, NeuralNetwork referenceNetwork, int repeats, int randomSeed) {
		NetworkInstance[] results = new NetworkInstance[workers];
		float[] newLearningRate = new float[workers];

		Task[] tasks = Enumerable.Range(0, workers).Select(async i => {
			NeuralNetwork network = GenerateMutation(referenceNetwork, mutationStrength);
			Trainer trainer = CreateTrainer(network, learningRate);
			Tester tester = CreateTester(network);
			Random localRandom = new Random(randomSeed);

			results[i] = new NetworkInstance(network, await GrowthProcess(trainer, repeats, localRandom, tester));
			newLearningRate[i] = trainer.LearningRate;
		}).ToArray();
			
			
		await Task.WhenAll(tasks);
			
		float averageLearningRate = newLearningRate.Average();
			
		return (results, averageLearningRate);
	}

	private NeuralNetwork GenerateMutation(NeuralNetwork network, float mutationStrength) {
		NeuralNetwork newNetwork = network.Copy();

		for (int i = 1; i < newNetwork.LayerAmount; i++) {
			Layer layer = (Layer)newNetwork[i];
				
			ITensor weightsDelta = GenerateOffset(newNetwork.Backend, layer.Weights, mutationStrength);
			ITensor biasDelta = GenerateOffset(newNetwork.Backend, layer.Bias, mutationStrength);
				
			newNetwork.Backend.Adjust(weightsDelta, biasDelta, 1, layer.Weights, layer.WeightsT, layer.Bias);
		}

		return newNetwork;
	}

	private ITensor GenerateOffset(IBackend backend, ITensor reference, float mutationStrength) {
		float[] data = new float[reference.Length];

		for (int i = 0; i < data.Length; i++) {
			data[i] = GuassianNoise(0, mutationStrength);
		}
			
		return backend.CreateTensor(data, reference.Shape.ToArray());
	}

	private float GuassianNoise(float mean = 0, float stdDev = 0.1f) {
		float u1 = 1 - random.NextSingle();
		float u2 = 1 - random.NextSingle();

		float randStdNormal = (float)(Math.Sqrt(-2.0f * Math.Log(u1)) * Math.Sin(2 * Math.PI * u2));
			
		return mean + stdDev * randStdNormal;
	}
}
	
internal class NetworkGrower(DataBatch<float> trainingData, DataBatch<float> testingData, bool isSilent = false) {
	public async Task<(NetworkInstance, float)> Grow(int workers, float learningRate, NetworkInstance referenceNetwork, int repeats, int randomSeed) {
		DateTime deltaTime = DateTime.Now;
		if (!isSilent) Console.WriteLine($"[Network Grower] Begun training {workers} networks.");
			
		(NetworkInstance[] newNetworks, float newLearningRate) = await TrainInParallel(workers, learningRate, referenceNetwork.Network, repeats, randomSeed);
		NetworkInstance bestNetwork = FindBestNetworkAndDiscardRest(referenceNetwork, newNetworks);

		if (!isSilent)
			Console.WriteLine(
				$"[Network Grower] Best Accuracy over {workers} workers: {bestNetwork.Accuracy} | " +
				$"Average Learning Rate: {learningRate} | " +
				$"Time elapsed: {DateTime.Now - deltaTime}");
			
		return (bestNetwork, newLearningRate);
	}
		
	private async Task<(NetworkInstance[], float)> TrainInParallel(int workers, float learningRate, NeuralNetwork referenceNetwork, int repeats, int randomSeed) {
		NetworkInstance[] results = new NetworkInstance[workers];
		float[] newLearningRate = new float[workers];

		Task[] tasks = Enumerable.Range(0, workers).Select(async i => {
			NeuralNetwork network = referenceNetwork.Copy();
			Trainer trainer = CreateTrainer(network, learningRate);
			Tester tester = CreateTester(network);
			Random random = new Random(randomSeed);

			results[i] = new NetworkInstance(network, await GrowthProcess(trainer, repeats, random, tester));
			newLearningRate[i] = trainer.LearningRate;
		}).ToArray();
			
			
		await Task.WhenAll(tasks);
			
		float averageLearningRate = newLearningRate.Average();
			
		return (results, averageLearningRate);
	}

	protected static NetworkInstance FindBestNetworkAndDiscardRest(NetworkInstance ogNetwork, NetworkInstance[] instances) {
		int max = 0;
		for (int i = 1; i < instances.Length; i++) {
			if (instances[i].Accuracy > instances[max].Accuracy) {
				instances[max].Dispose();
				max = i;
			} else {
				instances[i].Dispose();
			}
		}

		if (instances[max].Accuracy > ogNetwork.Accuracy) {
			ogNetwork.Dispose();
			return instances[max];
		}
			
		instances[max].Dispose();
		return ogNetwork;
	}

	protected async Task<int> GrowthProcess(Trainer trainer, int repeats, Random random, Tester tester) {
		await trainer.Train(trainingData, repeats, random);
		return await tester.Test(testingData);
	}

	protected static Trainer CreateTrainer(NeuralNetwork network, float learningRate) {
		return new Trainer(
			network,
			2500,
			10000,
			new TrainerLearningInfo(learningRate, 0.9f, 5),
			true
		);
	}
	public static Tester CreateTester(NeuralNetwork network) {
		return new Tester(
			network,
			10000,
			true
		);
	}
}