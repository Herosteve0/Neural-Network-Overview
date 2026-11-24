namespace NeuralNetwork.Core.Data;

/**
 * Allows flexible control over the Learning Rate.
 */
public class LearningRate {
    /**
     * Learning Rate with no decay.
     * <param name="learningRate">
     * The Learning Rate value.
     * </param>
     */
    public LearningRate(float learningRate) {
        ValidateLearningRate(learningRate);
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
    public LearningRate(float learningRate, float decayMultiplier, int decayPatience) {
        ValidateLearningRate(learningRate);
        if (!float.IsFinite(decayMultiplier) || decayMultiplier is < 0f or > 1f)
            throw new ArgumentOutOfRangeException(nameof(decayMultiplier), "Decay multiplier must be in range [0, 1].");
        if (decayPatience <= 0)
            throw new ArgumentOutOfRangeException(nameof(decayPatience), "Decay patience must be positive.");
        
        BaseLearningRate = learningRate;
        ResetLearningRate();
        
        _hasDecay = true;
            
        _decayMultiplier = decayMultiplier;
        _decayPatience = decayPatience;
    }

    private float BaseLearningRate { get; }
    private float _learningRate;

    private readonly bool _hasDecay;
    private readonly float _decayMultiplier;
    private readonly int _decayPatience;

    private int _patienceCounter;
    private float _minLoss = float.PositiveInfinity;

    private static void ValidateLearningRate(float learningRate) {
        if (!float.IsFinite(learningRate) || learningRate < 0f)
            throw new ArgumentOutOfRangeException(nameof(learningRate), "The learning rate must be positive.");
    }

    public static implicit operator float(LearningRate info) {
        return info?._learningRate ?? 0f;
    }
    
    /*
     * For more info, paste this in https://www.desmos.com/calculator
     * f\left(x\right)=\frac{\ln\left(c\left(1-x\right)+1\right)}{10\ln\left(c+1\right)}\left\{0\le x\le1\right\}
     */
    public static float SetFromAccuracy(float accuracy, float sharpness = 0f) {
        if (!float.IsFinite(accuracy) || accuracy is < 0f or > 1f)
            throw new ArgumentOutOfRangeException(nameof(accuracy), "Accuracy must be in range [0, 1].");
        if (!float.IsFinite(sharpness) || sharpness < 0f)
            throw new ArgumentOutOfRangeException(nameof(sharpness), "The sharpness must be greater or equal to 0.");
        
        if (sharpness == 0f) {
            return (1 - accuracy) / 10f;
        }
        return (float)(Math.Log(1 + sharpness * (1 - accuracy)) / Math.Log(1 + sharpness)) / 10f;
    }

    public bool HasDecayed => _learningRate == 0f;

    
    public void SetLearningRate(float learningRate) {
        ValidateLearningRate(learningRate);
        _learningRate = learningRate;
        _minLoss = float.PositiveInfinity;
        _patienceCounter = 0;
    }

    public LearningRate Copy() {
        return _hasDecay
            ? new LearningRate(BaseLearningRate, _decayMultiplier, _decayPatience)
            : new LearningRate(BaseLearningRate);
    }
    
    public void ResetLearningRate() {
        _learningRate = BaseLearningRate;
        _minLoss = float.PositiveInfinity;
        _patienceCounter = 0;
    }
        
    public void CheckLoss(float loss) {
        if (!float.IsFinite(loss))
            throw new ArgumentOutOfRangeException(nameof(loss), "Loss isn't finite.");
        if (!_hasDecay) return;
            
        if (loss < _minLoss) {
            _minLoss = loss;
            _patienceCounter = 0;
            return;
        }
            
        _patienceCounter++;
        if (_patienceCounter < _decayPatience) return;
            
        _patienceCounter = 0;
        ApplyDecay();
    }

    private void ApplyDecay() {
        _learningRate *= _decayMultiplier;
    }
}