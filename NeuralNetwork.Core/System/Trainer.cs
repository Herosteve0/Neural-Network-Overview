using NeuralNetwork.Core.System.Data;

namespace NeuralNetwork.Core.System;

public class TrainerTracker(int breathTicks) : ScannerTracker(breathTicks) {
    public float LearningRate { get; internal set; }
    public float Loss { get; internal set; }
    public int Accuracy { get; internal set; }
    
    public float AverageLoss => CurrentProgress == 0f ? 0f : Loss / CurrentProgress;
    public float AverageAccuracy => CurrentProgress == 0 ? 0 : 100f * Accuracy / CurrentProgress;

    public override void BeginIteration() {
        base.BeginIteration();
        
        Accuracy = 0;
        Loss = 0f;
    }
}

/**
 * Allows flexible control over the Learning Rate.
 */
public class LearningRateManager {
    /**
     * Learning Rate with no decay.
     * <param name="learningRate">
     * The Learning Rate value.
     * </param>
     */
    public LearningRateManager(float learningRate) {
        BaseLearningRate = learningRate;
        ResetLearningRate();

        _hasDecay = false;
    }
    /**
     * <param name="learningRate">
     * The Learning Rate value.
     * </param>
     * <param name="decayMultiplier">
     * The scalar of the learning rate.
     * </param>
     * <param name="decayPatience">
     * The amount of high losses in a row required to decay the learning rate.
     * </param>
     */
    public LearningRateManager(float learningRate, float decayMultiplier, int decayPatience) : this(learningRate) {
        _hasDecay = true;
            
        _decayMultiplier = decayMultiplier;
        _decayPatience = decayPatience;
    }

    private float BaseLearningRate { get; }
    private float LearningRate { get; set; }

    private readonly bool _hasDecay;
    private readonly float _decayMultiplier;
    private readonly float _decayPatience;

    private int _patienceCounter;
    private float _minLoss = float.PositiveInfinity;

    public static implicit operator float(LearningRateManager info) {
        if (info == null) return 0f;
        return info.LearningRate;
    }

    public bool HasDecayed => LearningRate == 0f;

    public void SetLearningRate(float learningRate) {
        LearningRate = learningRate;
    }
    public void ResetLearningRate() {
        LearningRate = BaseLearningRate;
    }
        
    public void CheckLoss(float loss) {
        if (!_hasDecay) return;
            
        if (loss < _minLoss) {
            _minLoss = loss;
            _patienceCounter = 0;
            return;
        }
            
        _patienceCounter++;
        if (_patienceCounter <= _decayPatience) return;
            
        _patienceCounter = 0;
        ApplyDecay();
    }

    public void ApplyDecay() {
        LearningRate *= _decayMultiplier;
    }
}

/**
 * A Network Scanner used to train networks.
 * <param name="batchSize">
 * The amount of data it trains over before adjusting the network's parameters.
 * </param>
 * <param name="breathTicks">
 * How many examples it executes before taking a breath. (aka, sleeping for a millisecond) 
 * </param>
 * <param name="learningInfo">
 * The learning rate manager of the trainer.
 * </param>
 * <param name="multiThread">
 * Whether it trains using threads. Recommended for Cpu Backends.
 * </param>
 */
public class Trainer(
    int batchSize,
    int breathTicks,
    LearningRateManager learningInfo,
    bool multiThread = false
) : NetworkScanner<TrainerTracker>(batchSize, breathTicks) {
    public async Task Train(Network network, Random random) {
        learningInfo.ResetLearningRate();
        await IterateNetwork(network, random);
    }

    public async Task Train(Network network, Random random, float learningRate) {
        learningInfo.SetLearningRate(learningRate);
        await IterateNetwork(network, random);
    }

    protected override void OnBreath() {
        Tracker.LearningRate = learningInfo;
    }

    protected override void Process(Network network, DataBatch<float> trainingData) {
        float scale = learningInfo / trainingData.Size;

        IBackend backend = network.Backend;
        
        ISingleValue loss = backend.CreateValue(0);
        ISingleValue accuracy = backend.CreateValue(0);

        if (multiThread) {
            ExecuteMultiThread(network, trainingData, scale, loss, accuracy);
        } else {
            Execute(network, trainingData, scale, loss, accuracy);
        }

        float lossValue = backend.ReadSingle(loss);
        loss.Dispose();
        
        int accuracyValue = (int)backend.ReadSingle(accuracy);
        accuracy.Dispose();

        learningInfo.CheckLoss(lossValue);
        if (learningInfo.HasDecayed) {
            ForceStop();
        }

        Tracker.Accuracy += accuracyValue;
        Tracker.Loss += lossValue;
    }

    private void Execute(Network network, DataBatch<float> dataBatch, float scale, ISingleValue loss, ISingleValue accuracy) {
        ExecuteSingle(
            dataBatch,
            () => new VirtualNetwork(network),
            (local, data) => TrainVirtualNetwork(network, local, data),
            (local) => {
                local.IterateAdjust(network, scale, loss, accuracy);
                local.Dispose();
            }
        );
    }
    
    private void ExecuteMultiThread(Network network, DataBatch<float> dataBatch, float scale, ISingleValue loss, ISingleValue accuracy) {
        ExecuteParallel(dataBatch,
            () => new VirtualNetwork(network),
            (local, data) => TrainVirtualNetwork(network, local, data),
            (local, lockObj) => {
                lock (lockObj) {
                    local.IterateAdjust(network, scale, loss, accuracy);
                }
                local.Dispose();
            }
        );
    }

    private static void TrainVirtualNetwork(Network network, VirtualNetwork virtualNetwork, DataSample<float> trainingData) {
        network.Calculate(trainingData, virtualNetwork);
        // all layers of the network have the values we want (input, value, activation)
        
        virtualNetwork.IterateBackward(network);
    }
}