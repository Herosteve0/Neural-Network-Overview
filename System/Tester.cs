using System.Drawing;
using Neurtal_Network_Overview.System.Data;

namespace Neurtal_Network_Overview.System;

public class Tester : NetworkScanner {
    private const int DelayTicks = 5000;
        
    public Tester(NeuralNetwork network, int batchSize, bool isSilent = false) : base("Tester", network, batchSize, DelayTicks, isSilent) { }
        
    private List<DataGuess<float>> _mistakes;

    public async Task<int> Test(DataBatch<float> dataBatch, bool showMistakes = false) {
        _mistakes = [];
            
        await Scan(dataBatch);
            
        int testingAccuracy = dataBatch.Size - _mistakes.Count;
        if (!IsSilent) {
            Console.WriteLine(
                $"Testing completed with {(double)testingAccuracy / dataBatch.Size * 100}% accuracy. [{testingAccuracy}/{dataBatch.Size}]");
        }

        if (showMistakes) {
            Console.WriteLine("Creating images (takes a bit...)");
            DrawImages(_mistakes.ToArray());
        }

        return testingAccuracy;
    }

    protected override void Process(DataBatch<float> testingData) {
        if (ProgramExecutor.MultiThread) {
            ExecuteMultiThread(testingData);
        }
        else {
            Execute(testingData);
        }
    }

    private void Execute(DataBatch<float> dataBatch) {
        VirtualNetwork network = new VirtualNetwork(Network);

        foreach (DataSample<float> testingData in dataBatch.Data) {
            TestingCalculations(testingData, network, _mistakes);
        }

        network.Dispose();
    }

    private void ExecuteMultiThread(DataBatch<float> dataBatch) {
        object lockObj = new object();

        Parallel.For(
            0L, dataBatch.Size,

            () => new LocalThread(Network),
            (i, state, local) => {
                DataSample<float> data = dataBatch.Data[i];
                    
                TestingCalculations(data, local.VirtualNetwork, local.Wrongs);

                return local;
            },

            local => {
                lock (lockObj) {
                    _mistakes.AddRange(local.Wrongs);
                }
                local.VirtualNetwork.Dispose();
            }

        );
    }
    private class LocalThread(NeuralNetwork network) {
        public readonly VirtualNetwork VirtualNetwork = new(network);
        public readonly List<DataGuess<float>> Wrongs = [];
    }

    private void TestingCalculations(DataSample<float> testingData, VirtualNetwork network, List<DataGuess<float>> wrongs) {
        Network.Calculate(testingData, network);
        
        if (!network.IsCorrect) {
            wrongs.Add(network.RawGuess);
        }
    }
        
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
    
        
    public static void DrawWeights(NeuralNetwork network){
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

        const float mult = 5f;

        Bitmap image = new Bitmap(width, height);

        float max = data.Max();
        float sum = data.Sum(value => (float)Math.Exp(value - max));
        
        int i = 0;
        for (int y = 0; y < height; y++) { 
            for (int x = 0; x < width; x++) {
                float value = data[i++];
                bool neg = value < 0;

                float color = (float)Math.Exp(value - max) / sum * mult * 255f;

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