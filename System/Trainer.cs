using Neurtal_Network_Overview.System.Data;

namespace Neurtal_Network_Overview.System;

public class TrainerLearningInfo {
    public TrainerLearningInfo(float learningRate) {
        BaseLearningRate = learningRate;
        ResetLearningRate();

        _hasDecay = false;
    }
    public TrainerLearningInfo(float learningRate, float decayMultiplier, int decayPatience) : this(learningRate) {
        _hasDecay = true;
            
        _decayMultiplier = decayMultiplier;
        _decayPatience = decayPatience;
    }

    private float BaseLearningRate { get; }
    public float LearningRate { get; private set; }

    private readonly bool _hasDecay;
    private readonly float _decayMultiplier;
    private readonly float _decayPatience;

    private int _patienceCounter = 0;
    private float _minLoss = float.PositiveInfinity;

    public static implicit operator float(TrainerLearningInfo info) {
        if (info == null) return 0;
        return info.LearningRate;
    }

    public bool HasDecayed => LearningRate == 0f;
        
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
    
public class Trainer : NetworkScanner {
    public Trainer(NeuralNetwork network, int batchSize, int delayTicks, TrainerLearningInfo learningInfo, bool isSilent = false) : base("Trainer", network, batchSize, delayTicks, isSilent) {
        _learningRate = learningInfo;
    }

    private readonly TrainerLearningInfo _learningRate;
        
    private float _lossMemory;
    private int _lossCount;

    public float LearningRate => _learningRate;

    public async Task Train(DataBatch<float> dataBatch, int repeats, Random random) {
        _lossMemory = 0f;
        _lossCount = 0;
        await Scan(dataBatch, repeats, random);
    }

    protected override void BreathFunction() {
        if (!IsSilent) {
            Console.WriteLine(
                $"Training Loss: {_lossMemory / _lossCount} | Learning Rate: {_learningRate.LearningRate}");
        }
        _lossMemory = 0f;
        _lossCount = 0;
    }

    protected override void Process(DataBatch<float> trainingData) {
        float scale = _learningRate / trainingData.Size;

        ISingleValue loss = Backend.CreateValue(0);

        if (ProgramExecutor.MultiThread) {
            ExecuteMultiThread(trainingData, scale, loss);
        } else {
            Execute(trainingData, scale, loss);
        }

        float averageLoss = Backend.ReadSingle(loss);
        loss.Dispose();
        
        _learningRate.CheckLoss(averageLoss);
        if (_learningRate.HasDecayed) {
            ForceStop("LearningRate Decayed.");
        }

        _lossMemory += averageLoss;
        _lossCount++;
    }

    private void Execute(DataBatch<float> dataBatch, float scale, ISingleValue loss) {
        VirtualNetwork network = new VirtualNetwork(Network);

        foreach (DataSample<float> trainingData in dataBatch.Data) {
            TrainVirtualNetwork(network, trainingData);
        }

        network.IterateAdjust(Network, scale, loss);
        network.Dispose();
    }

    private void ExecuteMultiThread(DataBatch<float> dataBatch, float scale, ISingleValue loss) {
        object lockObj = new();
        Parallel.For(0L, dataBatch.Size, () => new VirtualNetwork(Network), (i, state, local) => {
            DataSample<float> data = dataBatch[(int)i];

            TrainVirtualNetwork(local, data);

            return local;
        }, local => {
            lock (lockObj) {
                local.IterateAdjust(Network, scale, loss);
                local.Dispose();
            }
        });
    }

    private void TrainVirtualNetwork(VirtualNetwork network, DataSample<float> trainingData) {
        Network.Calculate(trainingData, network);
        // all layers of the network have the values we want (input, value, activation)
        
        network.IterateBackward(Network);
    }
}