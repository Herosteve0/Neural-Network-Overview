using System.Drawing;
using NeuralNetwork.Core.Backend;
using NeuralNetwork.Core.Data;
using NeuralNetwork.Core.Data.SingleValues;
using NeuralNetwork.Core.NetworkFormats;
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
            DrawImages(Tracker.GetMistakes());
        }

        return (float)_iterationCorrect / DataSize;
    }

    protected override void OnNewSample() {
        base.OnNewSample();

        _iterationCorrect = 0;
    }

    protected override void Process(Network network, DataBatch<float> testingData) {
        IBackend backend = network.Backend;
        ISingleValue<int> correct = backend.CreateValue(0);

        if (multiThread) {
            ExecuteMultiThread(network, testingData, correct);
        }
        else {
            Execute(network, testingData, correct);
        }

        int correctValue = backend.ReadSingle(correct);
        correct.Dispose();

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
                local.VirtualNetwork.Dispose();
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

                local.VirtualNetwork.Dispose();
            }
        );
    }

    private class LocalThread(Network network, int bufferSize) {
        public readonly VirtualNetwork VirtualNetwork = new(network, bufferSize);
        public readonly List<DataGuess<float>> Wrongs = [];
    }

    private void TestingCalculations(Network network, DataBatch<float> testingBatch, VirtualNetwork virtualNetwork, List<DataGuess<float>> wrongs) {
        virtualNetwork.LoadBuffer(testingBatch);
        
        virtualNetwork.ProcessFullBuffer(() => {
            virtualNetwork.CompleteForwardPass(network);
        });
        
        bool[] isCorrect = virtualNetwork.IsCorrect;

        for (int i = 0; i < BufferSize; i++)
            if (!isCorrect[i]) {
                wrongs.Add(virtualNetwork.RawGuess(i));
            }
    }

    // Visualizing Data
        
    public static void DrawImages(DataGuess<float>[] wrongs) {
        const string filepath = "../../../WrongGuesses";

        if (Path.Exists(filepath)) Directory.Delete(filepath, true);
        Directory.CreateDirectory(filepath);

        for (int i = 0; i < wrongs.Length; i++) {
            DataSample<float> data = wrongs[i].Data;
            int guess = wrongs[i].Prediction;
            
            CreateImage(data, $"{filepath}/{i}. {guess} instead of {data.Label}.png");
        }
    }
    private static void CreateImage(DataSample<float> data, string filepath) {
        const int width = 28;
        const int height = 28;

        Bitmap image = new Bitmap(width, height);

        int i = 0;
        for (int y = 0; y < height; y++) { 
            for (int x = 0; x < width; x++) {
                byte value = (byte)(Math.Clamp((int)(data.Values[i++] * 255f), 0, 255));
                image.SetPixel(x, y, Color.FromArgb(value, value, value));
            }
        }

        image.Save(filepath);
    }
        
    public static void DrawWeights(Network network){
        const string filepath = "../../../WeightImages";

        if (Path.Exists(filepath)) Directory.Delete(filepath, true);
        Directory.CreateDirectory(filepath);

        ReadOnlySpan<float> weights = network.GetData().weights[0];
        int step = network.LayerLength[0];
        int count = 0;
        for (int i = 0; i < weights.Length; i += step) {
            CreateWeightImage(weights.Slice(i, step).ToArray(), $"{filepath}/Weight-{count++}.png");
        }
    }
    private static void CreateWeightImage(float[] data, string filepath) {
        const int width = 28;
        const int height = 28;

        const float mult = 255f * 10f;

        Bitmap image = new Bitmap(width, height);

        float max = data.Max();
        float sum = data.Sum(value => (float)Math.Exp(value - max));
        
        int i = 0;
        for (int y = 0; y < height; y++) { 
            for (int x = 0; x < width; x++) {
                float value = data[i++];
                bool neg = value < 0;

                float color = (float)Math.Exp(value - max) / sum * mult;

                byte b = (byte)color;

                image.SetPixel(x, y,
                    neg
                        ? Color.FromArgb(b, 0, 0) 
                        : Color.FromArgb(0, 0, b)
                );
            }
        }

        image.Save(filepath);
    }
}