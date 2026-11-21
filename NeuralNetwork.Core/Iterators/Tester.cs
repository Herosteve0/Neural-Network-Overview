using NeuralNetwork.Core.Backend;
using NeuralNetwork.Core.Data;
using NeuralNetwork.Core.Data.SingleValues;
using NeuralNetwork.Core.NetworkFormats;
using NeuralNetwork.Core.NetworkFormats.Layers.State;
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

    /**
     * <returns>
     * the accuracy of the iteration as a float in the range [0, 1].
     * </returns>
     */
    public async Task<float> Test(Network network, bool showMistakes = false) {
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
            ExecuteMultiThread(network, testingData, correct);
        } else {
            Execute(network, testingData, correct);
        }

        int correctValue = backend.ReadSingle(correct);
        _iterationCorrect += correctValue;
    }

    private void Execute(Network network, DataBatch<float> dataBatch, ISingleValue<int> correct) {
        ExecuteSingle(
            dataBatch,
            () => new LocalThread(network, BufferSize),
            (local, data) => TestingCalculations(network, data, local.VirtualNetwork, local.Wrongs),
            (local) => {
                local.VirtualNetwork.AdjustCorrect(correct);
                Tracker.Update(local.Wrongs);
                local.Dispose();
            }
        );
    }

    private void ExecuteMultiThread(Network network, DataBatch<float> dataBatch, ISingleValue<int> correct) {
        int processors = Environment.ProcessorCount;
        
        ExecuteParallel(
            dataBatch,
            processors,
            () => new LocalThread(network, BufferSize),
            (local, buffer) => TestingCalculations(network, buffer, local.VirtualNetwork, local.Wrongs),
            (local, lockObj) => {
                lock (lockObj) {
                    local.VirtualNetwork.AdjustCorrect(correct);
                    Tracker.Update(local.Wrongs);
                }

                local.Dispose();
            }
        );
    }

    private class LocalThread(Network network, int bufferSize) : IDisposable {
        public readonly VirtualNetwork VirtualNetwork = new(network, bufferSize);
        public readonly List<DataGuess<float>> Wrongs = [];
        
        public void Dispose() => VirtualNetwork.Dispose();
    }

    private static void TestingCalculations(Network network, DataBatch<float> testingBatch, VirtualNetwork virtualNetwork,
        List<DataGuess<float>> wrongs) {
        virtualNetwork.LoadBuffer(testingBatch);

        virtualNetwork.ProcessFullBuffer(() => { virtualNetwork.CompleteForwardPass(network, PassMode.Inference); });

        bool[] isCorrect = virtualNetwork.IsCorrect;

        for (int i = 0; i < isCorrect.Length; i++) {
            if (!isCorrect[i]) {
                wrongs.Add(virtualNetwork.RawGuess(i));
            }
        }
    }
}