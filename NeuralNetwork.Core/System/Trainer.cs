using System.Globalization;
using NeuralNetwork.Core.System.Backend;
using NeuralNetwork.Core.System.Data;
using NeuralNetwork.Core.System.Trackers;

namespace NeuralNetwork.Core.System;

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
        return info?.LearningRate ?? 0f;
    }
    
    /*
     * For more info, paste this in https://www.desmos.com/calculator
     * f\left(x\right)=\frac{\ln\left(c\left(1-x\right)+1\right)}{10\ln\left(c+1\right)}\left\{0\le x\le1\right\}
     */
    public static float SetFromAccuracy(float accuracy, float sharpness = 0f) {
        if (sharpness == 0f) {
            return (1 - accuracy) / 10f;
        }
        return (float)(Math.Log(1 + sharpness * (1 - accuracy)) / Math.Log(1 + sharpness)) / 10f;
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
    private float _lossGain;
    private int _correctGain;
    
    public async Task<TimeSpan> Train(Network network, Random random) {
        learningInfo.ResetLearningRate();
        return await IterateNetwork(network, random);
    }

    public async Task<TimeSpan> Train(Network network, Random random, float learningRate) {
        learningInfo.SetLearningRate(learningRate);
        return await IterateNetwork(network, random);
    }

    protected override void OnBreath() {
        base.OnFinish();
        
        Tracker.Update(learningInfo, _lossGain, _correctGain);
        _lossGain = 0f;
        _correctGain = 0;
    }

    protected override void OnFinish() {
        base.OnFinish();
        
        Tracker.Update(learningInfo, _lossGain, _correctGain);
        _lossGain = 0f;
        _correctGain = 0;
    }

    protected override void Process(Network network, DataBatch<float> trainingData) {
        float scale = learningInfo / trainingData.Size;

        IBackend backend = network.Backend;
        
        IValueFloat loss = backend.CreateValue(0f);
        IValueInt correct = backend.CreateValue(0);

        if (multiThread) {
            ExecuteMultiThread(network, trainingData, scale, loss, correct);
        } else {
            Execute(network, trainingData, scale, loss, correct);
        }

        float lossValue = backend.ReadSingle(loss);
        loss.Dispose();
        
        int correctValue = backend.ReadSingle(correct);
        correct.Dispose();

        learningInfo.CheckLoss(lossValue);
        if (learningInfo.HasDecayed) {
            ForceStop();
        }

        _correctGain += correctValue;
        _lossGain += lossValue;
    }

    private void Execute(Network network, DataBatch<float> dataBatch, float scale, IValueFloat loss, IValueInt correct) {
        ExecuteSingle(
            dataBatch,
            () => new VirtualNetwork(network),
            (local, data) => TrainVirtualNetwork(network, local, data),
            (local) => {
                local.IterateAdjust(network, scale);
                local.AdjustCorrect(correct);
                local.AdjustLoss(loss);
                local.Dispose();
            }
        );
    }
    
    private void ExecuteMultiThread(Network network, DataBatch<float> dataBatch, float scale, IValueFloat loss, IValueInt correct) {
        ExecuteParallel(dataBatch,
            () => new VirtualNetwork(network),
            (local, data) => TrainVirtualNetwork(network, local, data),
            (local, lockObj) => {
                lock (lockObj) {
                    local.IterateAdjust(network, scale);
                    local.AdjustCorrect(correct);
                    local.AdjustLoss(loss);
                }
                local.Dispose();
            }
        );
    }

    private static void TrainVirtualNetwork(Network network, VirtualNetwork virtualNetwork, DataSample<float> trainingData) {
        network.Calculate(trainingData, virtualNetwork);
        virtualNetwork.IterateBackward(network);
    }
}