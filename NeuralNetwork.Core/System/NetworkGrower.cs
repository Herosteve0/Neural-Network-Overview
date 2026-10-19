using NeuralNetwork.Core.Data;

namespace NeuralNetwork.Core;

public class GrowerTracker {
    private readonly TrainerTracker[] _trainers;
    private readonly TesterTracker[] _testers;
    private readonly int _processors;

    public int Generations { get; private set; }
    public int CurrentGeneration { get; internal set; }
    
    public GrowerTracker(int processors, TrainerTracker[] trainers, TesterTracker[] testers) {
        _processors = processors;
        _trainers = trainers;
        _testers = testers;
    }

    public float AverageAccuracy => (float)_trainers.Select(tracker => tracker.AverageAccuracy).Average();
    public float AverageLoss => _trainers.Select(tracker => tracker.AverageLoss).Average();

    public int TotalExamples => _trainers[0].TotalExamples + _testers[0].TotalExamples;
    public int Progress => (int)(_trainers.Select(tracker => tracker.Progress).Average() + _testers.Select(tracker => tracker.Progress).Average());

    public void Begin(int generations) {
        Generations = generations;
        CurrentGeneration = 0;
    }

    public void BeginGeneration() { }
}

/**
 * Allows flexible control over the mutation strength.
 */
public class MutationStrengthManager {
    /**
     * Mutation Strength with no decay.
     * <param name="mutationStrength">
     * The Mutation Strength value.
     * </param>
     */
    public MutationStrengthManager(float mutationStrength) {
        BaseMutationStrength = mutationStrength;
        ResetLearningRate();

        _hasDecay = false;
    }
    /**
     * <param name="mutationStrength">
     * The Mutation Strength value.
     * </param>
     * <param name="decayMultiplier">
     * The scalar of the learning rate.
     * </param>
     */
    public MutationStrengthManager(float mutationStrength, float decayMultiplier) : this(mutationStrength) {
        _hasDecay = true;
            
        _decayMultiplier = decayMultiplier;
    }
	
    /**
     * <returns>
     * a manager with Mutation Strength = 0f
     * </returns>
     */
    public static MutationStrengthManager None { get; } = new MutationStrengthManager(0f);
	
    private float BaseMutationStrength { get; }
    private float MutationStrength { get; set; }

    private readonly bool _hasDecay;
    private readonly float _decayMultiplier;

    public static implicit operator float(MutationStrengthManager info) {
        if (info == null) return 0f;
        return info.MutationStrength;
    }

    public bool HasDecayed => MutationStrength == 0f;
	
    public void SetLearningRate(float mutationStrength) {
        MutationStrength = mutationStrength;
    }
    public void ResetLearningRate() {
        MutationStrength = BaseMutationStrength;
    }
	
    public void ApplyDecay() {
        if (!_hasDecay) return;
        MutationStrength *= _decayMultiplier;
    }

}

/**
 * <param name="DefaultLearningRate">
 * The learning rate the grower falls back to if the network has no set accuracy. If it does, the accuracy is equal to (1 - accuracy%) / 10
 * </param>
 * <param name="RngSeed">
 * The base seed used for the Random class of each network. (seed = RndSeed + processor#)
 * </param>
 * <param name="MutationStrength">
 * The mutation info of the grower.
 * </param>
 */
public record NetworkGrowerInfo(float DefaultLearningRate, int RngSeed, MutationStrengthManager MutationStrength) {
    public void DecayMutationStrength() {
        MutationStrength.ApplyDecay();
    }
}

/**
 * A class that can train multiple instances or mutations of the same network on different threats.
 */
public class NetworkGrower {
    private readonly int _processors;
    
    private readonly Trainer[] _trainers;
    private readonly Tester[] _testers;

    public readonly GrowerTracker Tracker;

    /**
     * <param name="processors">
     * The amount of instances that will be trained. Clamps between 1 and the processor count of the system.
     * </param>
     * <param name="trainingData">
     * The data batch for the training.
     * </param>
     * <param name="recycles">
     * The amount of times the training data will be recycled/reused.
     * </param>
     * <param name="testingData">
     * The data batch for the testing.
     * </param>
     */
    public NetworkGrower(int processors, DataBatch<float> trainingData, int recycles, DataBatch<float> testingData) {
        _processors = Math.Clamp(processors, 1, Environment.ProcessorCount);
        
        _trainers = new Trainer[_processors];
        for (int i = 0; i < _trainers.Length; i++) {
            _trainers[i] = CreateTrainer(trainingData, recycles, 0.085f);
        }
        
        _testers = new Tester[_processors];
        for (int i = 0; i < _testers.Length; i++) {
            _testers[i] = CreateTester(testingData);
        }

        Tracker = new GrowerTracker(
            _processors,
            _trainers.Select(trainer => trainer.Tracker).ToArray(),
            _testers.Select(tester => tester.Tracker).ToArray()
        );
    }

    private static Trainer CreateTrainer(DataBatch<float> dataBatch, int recycles, float learningRate) {
        Trainer trainer = new Trainer(
            2500,
            5000,
            new LearningRateManager(learningRate, 0.9f, 5)
        );
        trainer.LoadBatch(dataBatch, recycles);
        return trainer;
    }
    private static Tester CreateTester(DataBatch<float> dataBatch) {
        Tester tester = new Tester(
            10000
        );
        tester.LoadBatch(dataBatch);
        return tester;
    }

    /**
     * Grows multiple generations of networks and returns the best one.
     */
    public async Task<GrowableNetwork> GenTrain(GrowableNetwork network, int generations, NetworkGrowerInfo info) {
        Tracker.Begin(generations);
        
        for (int gen = 0; gen < generations; gen++) {
            network = await Grow(network, info);
            info.DecayMutationStrength();

            Tracker.CurrentGeneration++;
        }
        
        return network;
    }
    
    private async Task<GrowableNetwork> Grow(GrowableNetwork referenceNetwork, NetworkGrowerInfo info) {
        GrowableNetwork[] newNetworks = await TrainInParallel(
            _processors,
            referenceNetwork,
            info.RngSeed,
            info.DefaultLearningRate,
            info.MutationStrength
        );
        GrowableNetwork bestNetwork = FindBestNetworkAndDiscardRest(referenceNetwork, newNetworks);
        
        return bestNetwork;
    }
    
    private async Task<GrowableNetwork[]> TrainInParallel(
        int processes, GrowableNetwork referenceNetwork, int randomSeed, float baseLearningRate, float mutationStrength = 0f
    ) {
        GrowableNetwork[] results = new GrowableNetwork[processes];
        
        Task[] tasks = Enumerable.Range(0, processes).Select(async i => {
            Random random = new Random(unchecked(randomSeed + i));
            
            GrowableNetwork network = i == 0 ? referenceNetwork.Copy() : referenceNetwork.Mutate(random, mutationStrength);
            float learningRate = network.Accuracy == 0 ? baseLearningRate : (1f - network.Accuracy) / 10f;

            results[i] = await network.TrainAndEvaluate(
                _trainers[i], random, learningRate,
                _testers[i]
            );
        }).ToArray();
        
        await Task.WhenAll(tasks);
        
        return results;
    }
    
    private static GrowableNetwork FindBestNetworkAndDiscardRest(GrowableNetwork ogNetwork, GrowableNetwork[] instances) {
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

    public void ForceStop() {
        foreach (Trainer trainer in _trainers) {
            trainer.ForceStop();
        }
    }
}

public class GrowableNetworkTracker {
    public float Accuracy;
    
    public TrainerTracker TrainerTracker;
    public TesterTracker TesterTracker;

    public GrowableNetworkTracker(float accuracy = 0f) {
        Accuracy = accuracy;
    }
    
    
}

public class GrowableNetwork {
    public GrowableNetworkTracker Tracker;
    
    private NetworkInstance Instance { get; }
    private Network Network => Instance.Network;
    private IBackend Backend => Network.Backend;
    public float Accuracy => Instance.Accuracy;

    public GrowableNetwork(int[] layers, IBackend backend, Random random) : this(
            new Network(layers, backend, random), 0
        ) { }
    public GrowableNetwork(Network network, int accuracy) : this(
        new NetworkInstance(network, accuracy)
    ) { }
    public GrowableNetwork(NetworkInstance instance) {
        Instance = instance;

        Tracker = new GrowableNetworkTracker {
            Accuracy = Accuracy
        };
    }

    public static implicit operator NetworkInstance(GrowableNetwork network) {
        return network.Instance;
    }

    public GrowableNetwork Copy() {
        return new GrowableNetwork(Instance.Copy());
    }

    public void Dispose() {
        Instance.Dispose();
    }

    public async Task<GrowableNetwork> TrainAndEvaluate(
        Trainer trainer, Random random, float learningRate, Tester tester
    ) {
        Tracker.TrainerTracker = trainer.Tracker;
        Tracker.TesterTracker = tester.Tracker;
        
        trainer.Tracker.Reset();
        tester.Tracker.Reset();
        
        await trainer.Train(
            Network,
            random,
            learningRate
        );
        
        int accuracy = await tester.Test(Network);

        Tracker.TrainerTracker = null;
        Tracker.TesterTracker = null;
        
        return new GrowableNetwork(Network, accuracy);
    }

    public GrowableNetwork Mutate(Random random, float mutationStrength) {
        GrowableNetwork newNetwork = Copy();

        if (mutationStrength == 0f) return newNetwork;

        Network network = newNetwork.Network;
        IBackend backend = newNetwork.Backend;

        for (int i = 1; i < network.LayerAmount; i++) {
            Layer layer = (Layer)network[i];
				
            ITensor weightsDelta = GenerateOffset(backend, random, layer.Weights, mutationStrength);
            ITensor biasDelta = GenerateOffset(backend, random, layer.Bias, mutationStrength);
				
            backend.Adjust(weightsDelta, biasDelta, 1, layer.Weights, layer.WeightsT, layer.Bias);
            
            weightsDelta.Dispose();
            biasDelta.Dispose();
        }

        return newNetwork;
    }

    private static ITensor GenerateOffset(IBackend backend, Random random, ITensor reference, float mutationStrength) {
        float[] data = new float[reference.Length];

        for (int i = 0; i < data.Length; i++) {
            data[i] = GaussianNoise(random, 0, mutationStrength);
        }
			
        return backend.CreateTensor(data, reference.Shape.ToArray());
    }

    // Temporary
    private static float GaussianNoise(Random random, float mean = 0, float stdDev = 0.1f) {
        float u1 = 1 - random.NextSingle();
        float u2 = 1 - random.NextSingle();

        float randStdNormal = (float)(Math.Sqrt(-2.0f * Math.Log(u1)) * Math.Sin(2 * Math.PI * u2));
			
        return mean + stdDev * randStdNormal;
    }
}