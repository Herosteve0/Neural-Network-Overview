using System.Drawing;
using NeuralNetwork.Core.System.Data;

namespace NeuralNetwork.Core.System;

public class TesterTracker(int breathTicks) : ScannerTracker(breathTicks) {
    public List<DataGuess<float>> Mistakes;

    public int Accuracy => TotalExamples - Mistakes.Count;
    public float AccuracyPercent => Accuracy / (float)TotalExamples * 100;

    public override void Reset() {
        base.Reset();
        Mistakes = [];
    }

    public DataGuess<float>[] GetMistakes() {
        return Mistakes.ToArray();
    }
}

/**
 * A Network Scanner used to test networks.
 */
public class Tester(
    int batchSize,
    bool multiThread = false
) : NetworkScanner<TesterTracker>(batchSize, 5000) {
    public async Task<float> Test(Network network, bool showMistakes = false) {
        await IterateNetwork(network);
            
        float testingAccuracy = Tracker.AccuracyPercent;

        if (showMistakes) {
            Console.WriteLine("Creating images (takes a bit...)");
            DrawImages(Tracker.GetMistakes());
        }

        return testingAccuracy;
    }

    protected override void Process(Network network, DataBatch<float> testingData) {
        if (multiThread) {
            ExecuteMultiThread(network, testingData);
        } else {
            Execute(network, testingData);
        }
    }

    private void Execute(Network network, DataBatch<float> dataBatch) {
        ExecuteSingle(
            dataBatch,
            () => new VirtualNetwork(network),
            (local, data) => TestingCalculations(network, data, local, Tracker.Mistakes),
            (local) => local.Dispose()
        );
    }

    private void ExecuteMultiThread(Network network, DataBatch<float> dataBatch) {
        ExecuteParallel(
            dataBatch,
            () => new LocalThread(network),
            (local, data) => TestingCalculations(network, data, local.VirtualNetwork, local.Wrongs),
            (local, lockObj) => {
                lock (lockObj) {
                    Tracker.Mistakes.AddRange(local.Wrongs);
                }
                local.VirtualNetwork.Dispose();
            }
        );
    }
    private class LocalThread(Network network) {
        public readonly VirtualNetwork VirtualNetwork = new(network);
        public readonly List<DataGuess<float>> Wrongs = [];
    }

    private static void TestingCalculations(Network network, DataSample<float> testingData, VirtualNetwork virtualNetwork, List<DataGuess<float>> wrongs) {
        network.Calculate(testingData, virtualNetwork);
        
        if (!virtualNetwork.IsCorrect) {
            wrongs.Add(virtualNetwork.RawGuess);
        }
    }
    
    // Visualizing Data
        
    public static void DrawImages(DataGuess<float>[] wrongs){
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