using NeuralNetwork.Core.Backend;
using NeuralNetwork.Core.Data;
using NeuralNetwork.Core.Data.SingleValues;
using NeuralNetwork.Core.NetworkFormats;
using NeuralNetwork.Core.NetworkFormats.Layers.State;
using NeuralNetwork.Core.Trackers;

namespace NeuralNetwork.Core.Iterators;

/**
 * A Network Scanner used to train networks.
 * <param name="batchSize">
 * The amount of data it trains over before adjusting the network's parameters.
 * </param>
 * <param name="bufferSize">
 * The amount of data it loads into the virtual network at once.
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
    int bufferSize,
    int breathTicks,
    LearningRate learningInfo,
    bool multiThread = false
) : NetworkScanner<TrainerTracker>(batchSize, bufferSize, breathTicks) {
    private float _lossGain;
    private float _iterationLoss;
    private int _correctGain;
    
    public async Task<TimeSpan> Train(Network network, Random random) {
        learningInfo.ResetLearningRate();
        return await IterateNetwork(network, random);
    }

    public async Task<TimeSpan> Train(Network network, Random random, float learningRate) {
        learningInfo.SetLearningRate(learningRate);
        return await IterateNetwork(network, random);
    }

    protected override void OnNewSample() {
        base.OnNewSample();
        _iterationLoss = 0f;
    }

    protected override void OnEndSample() {
        base.OnEndSample();
        learningInfo.CheckLoss(_iterationLoss / DataSize);
        if (learningInfo.HasDecayed) {
            ForceStop();
        }
    }

    protected override void OnBreath() {
        base.OnBreath();
        
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
        
        ISingleValue<float> loss = backend.CreateValue(0f);
        ISingleValue<int> correct = backend.CreateValue(0);

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

    private void Execute(Network network, DataBatch<float> dataBatch, float scale, ISingleValue<float> loss, ISingleValue<int> correct) {
        ExecuteSingle(
            dataBatch,
            () => new VirtualNetwork(network, BufferSize),
            (local, buffer) => TrainVirtualNetwork(network, local, buffer),
            (local) => {
                local.IterateAdjust(network, scale);
                local.AdjustCorrect(correct);
                local.AdjustLoss(loss);
                local.Dispose();
            }
        );
    }
    
    private void ExecuteMultiThread(Network network, DataBatch<float> dataBatch, float scale, ISingleValue<float> loss, ISingleValue<int> correct) {
        int processors = Environment.ProcessorCount;
        
        ExecuteParallel(dataBatch,
            processors,
            () => new VirtualNetwork(network, BufferSize),
            (local, buffer) => TrainVirtualNetwork(network, local, buffer),
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

    private static void TrainVirtualNetwork(Network network, VirtualNetwork virtualNetwork, DataBatch<float> trainingBuffer) {
        virtualNetwork.LoadBuffer(trainingBuffer);
        
        virtualNetwork.ProcessFullBuffer(() => {
                virtualNetwork.CompleteForwardPass(network, PassMode.Training);
                virtualNetwork.IterateBackward(network);
            }
        );
    }
}