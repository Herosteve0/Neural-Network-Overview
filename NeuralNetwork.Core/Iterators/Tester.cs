using NeuralNetwork.Core.Backend;
using NeuralNetwork.Core.Data;
using NeuralNetwork.Core.Data.SingleValues;
using NeuralNetwork.Core.NetworkFormats.ExecutableNetworks;
using NeuralNetwork.Core.NetworkFormats.Networks;
using NeuralNetwork.Core.Others;
using NeuralNetwork.Core.Trackers;

namespace NeuralNetwork.Core.Iterators;

/**
 * A Network Scanner used to test networks.
 */
public class Tester(
    int batchSize,
    bool multiThread = false
) : NetworkScanner<TesterTracker>(batchSize, 1, 5000) {
    private int _iterationCorrect;
    private bool _showMistakes;

    /**
     * <returns>
     * the accuracy of the iteration as a float in the range [0, 1].
     * </returns>
     */
    public async Task<float> Test(Network network, bool showMistakes = false) {
        _showMistakes = showMistakes;
        
        await IterateNetwork(network);
        
        if (showMistakes) {
            Console.WriteLine("Creating images (takes a bit...)");
            DisplayManager.DrawImages(Tracker.GetMistakes());
        }

        return (float)_iterationCorrect / DataSize;
    }

    protected override void OnNewSample() {
        base.OnNewSample();

        _iterationCorrect = 0;
    }

    protected override void Process(Network network, DataBatch<float> testingData) {
        IBackend backend = network.Backend;
        
        using ISingleValue<int> correct = backend.CreateValue(0);

        if (multiThread) {
            ExecuteMultiThread(network, testingData, correct, _showMistakes);
        } else {
            Execute(network, testingData, correct, _showMistakes);
        }

        int correctValue = backend.Read(correct);
        _iterationCorrect += correctValue;
    }

    private void Execute(Network network, DataBatch<float> dataBatch, ISingleValue<int> correct, bool countMistakes = false) {
        ExecuteSingle(
            dataBatch,
            () => new LocalThread(network, BufferSize, countMistakes),
            (local, data) => TestingCalculations(network, data, local.InferenceNetwork, local.Mistakes),
            (local) => {
                local.InferenceNetwork.AdjustCorrect(correct);
                Tracker.Update(local.Mistakes);
                local.Dispose();
            }
        );
    }

    private void ExecuteMultiThread(Network network, DataBatch<float> dataBatch, ISingleValue<int> correct, bool countMistakes = false) {
        int processors = Environment.ProcessorCount;
        
        ExecuteParallel(
            dataBatch,
            processors,
            () => new LocalThread(network, BufferSize, countMistakes),
            (local, buffer) => TestingCalculations(network, buffer, local.InferenceNetwork, local.Mistakes),
            (local, lockObj) => {
                lock (lockObj) {
                    local.InferenceNetwork.AdjustCorrect(correct);
                    Tracker.Update(local.Mistakes);
                }

                local.Dispose();
            }
        );
    }

    private class LocalThread(Network network, int bufferSize, bool countMistakes = false) : IDisposable {
        public readonly InferenceNetwork InferenceNetwork = new(network, bufferSize);
        public readonly List<DataGuess<float>> Mistakes = countMistakes ? [] : null;

        public void Dispose() => InferenceNetwork.Dispose();
    }

    private static void TestingCalculations(Network network, DataBatch<float> testingBatch,
        InferenceNetwork inferenceNetwork, List<DataGuess<float>> mistakes) {
        inferenceNetwork.LoadBuffer(testingBatch);

        inferenceNetwork.ProcessFullBuffer(() => { inferenceNetwork.CompleteForwardPass(network); });

        if (mistakes == null) return;
        
        bool[] isCorrect = inferenceNetwork.IsCorrect;
        
        for (int i = 0; i < isCorrect.Length; i++) {
            if (!isCorrect[i]) {
                mistakes.Add(inferenceNetwork.RawGuess(i));
            }
        }
    }
}