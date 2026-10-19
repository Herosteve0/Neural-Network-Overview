using System.Windows;
using NeuralNetwork.Core;

namespace NeuralNetwork.UI;

public partial class MainWindow {

    private bool _isTraining;
    
    public MainWindow() {
        InitializeComponent();
        
        NetworkManager.DefaultInitialize();
    }

    private void ConsoleLog(string text) {
        LogList.Items.Add(text);
    }
    
    private void TrainingControlButton(object sender, RoutedEventArgs e) {
        LogList.Items.Clear();
        _isTraining = !_isTraining;
        
        ToggleInterface(_isTraining);

        if (_isTraining) {
            StartTraining();
        } else {
            StopTraining();
        }
    }

    private void ToggleInterface(bool state) {
        StatusText.Text = state ? "Training..." : "Idle";
        ConsoleLog("Training " + (state ? "started" : "stopped") + ".");
        ControlButton.Content = (state ? "Stop" : "Start") + " Training.";
    }
    
    private async void StartTraining() {
        try {
            IProgress<GrowerTracker> progress = new Progress<GrowerTracker>(UpdateInterface);
        
            await Task.Run(async () => {
                Task training = NetworkManager.GroupTraining(true);
                GrowerTracker tracker = NetworkManager.Grower.Tracker;

                while (!training.IsCompleted) {
                    progress.Report(tracker);
                    await Task.Delay(1000);
                }
            });

            ConsoleLog("Training completed.");
            _isTraining = false;
            ToggleInterface(false);
        } catch (Exception ex) {
            StatusText.Text = "Training failed.";
            ConsoleLog($"Error: {ex.Message}");
        }
    }

    private void StopTraining() {
        NetworkManager.ForceStop();
    }

    private void UpdateInterface(GrowerTracker tracker) {
        UpdateProgress(tracker);
        LossText.Text = $"Loss: {tracker.AverageLoss:F6}";
        AccuracyText.Text = $"Accuracy: {tracker.AverageAccuracy}";
    }

    private void UpdateProgress(GrowerTracker tracker) {
        int progress = tracker.Progress;
        float total = tracker.TotalExamples;
        
        int gen = tracker.CurrentGeneration;
        int totalGen = tracker.Generations;

        TotalProgress.Value = (progress / total + gen) / totalGen;
        TrainingProgress.Value = progress / total;
        TrainingProgressText.Text = $"Generation: {gen + 1} / {totalGen} | Samples trained over: {progress} / {total}";
    }
}