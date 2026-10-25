using System.Windows;
using NeuralNetwork.Core;
using NeuralNetwork.Core.System;
using NeuralNetwork.Core.Trackers;

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
            IProgress<GrowerTracker> progress = new Progress<GrowerTracker>(tracker => UpdateInterface(tracker.Snapshot()));
        
            await Task.Run(async () => {
                Task training = NetworkManager.GroupTraining(true);
                GrowerTracker tracker = NetworkManager.Grower.Tracker;

                while (!training.IsCompleted) {
                    progress.Report(tracker);
                    await Task.Delay(1000);
                }
            });

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

    private void UpdateInterface(GrowerProgressSnapshot snapshot) {
        UpdateProgress(snapshot);
        LossText.Text = $"Loss: {snapshot.Loss:F6}";
        AccuracyText.Text = $"Accuracy: {snapshot.Accuracy}%";
    }

    private void UpdateProgress(GrowerProgressSnapshot snapshot) {
        TotalProgress.Value = ((float)snapshot.Progress / snapshot.TotalExamples + snapshot.Generation) / snapshot.TotalGenerations;
        TrainingProgress.Value = (float)snapshot.Progress / snapshot.TotalExamples;
        TrainingProgressText.Text = $"Generation: {snapshot.Generation + 1} / {snapshot.TotalGenerations} | Samples trained over: {snapshot.Progress} / {snapshot.TotalExamples}";
    }
}