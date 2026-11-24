using System.Drawing;
using NeuralNetwork.Core.Data;
using NeuralNetwork.Core.NetworkFormats;
using NeuralNetwork.Core.NetworkFormats.Layers;

namespace NeuralNetwork.Core.Others;

public static class DisplayManager {
    private static readonly string WrongGuessPath = FileManager.WorkspacePath + "WrongGuesses";
    private static readonly string WeightImagePath = FileManager.WorkspacePath + "WeightImages";
    
    public static void DrawImages(DataGuess<float>[] wrongs) {
        string filepath = WrongGuessPath;

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

        if (data.Values is null || data.Values.Length != width * height)
            throw new ArgumentException($"Expected exactly {width * height} pixel values.", nameof(data));
        
        using Bitmap image = new Bitmap(width, height);
        
        int i = 0;
        for (int y = 0; y < height; y++) { 
            for (int x = 0; x < width; x++) {
                byte value = (byte)Math.Clamp((int)(data.Values[i++] * 255f), 0, 255);
                image.SetPixel(x, y, Color.FromArgb(value, value, value));
            }
        }

        image.Save(filepath);
    }
        
    public static void DrawWeights(Network network) {
        string filepath = WeightImagePath;

        if (Path.Exists(filepath)) Directory.Delete(filepath, true);
        Directory.CreateDirectory(filepath);

        FullyConnectedLayer firstDenseLayer = network.Layers.OfType<FullyConnectedLayer>().FirstOrDefault()
            ?? throw new InvalidOperationException($"The network does not contain a fully connected layer.");
        
        float[][] weights = firstDenseLayer.GetWeights();
        for (int i = 0; i < weights.Length; i++) {
            CreateWeightImage(weights[i], $"{filepath}/Weight-{i}.png");
        }
    }
    private static void CreateWeightImage(float[] data, string filepath) {
        const int width = 28;
        const int height = 28;
        
        const float mult = 255f * 10f;
        if (data is null || data.Length != width * height)
            throw new ArgumentException($"Expected exactly {width * height} weights for a {width}x{height} image.", nameof(data));
        
        using Bitmap image = new Bitmap(width, height);
        
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