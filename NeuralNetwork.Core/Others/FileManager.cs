namespace NeuralNetwork.Core.Others;

public static class FileManager {
    public static string WorkspacePath => GetWorkspacePath();

    private static string GetWorkspacePath() {
        string processPath = Environment.ProcessPath;
        if (processPath == null)
            throw new Exception("Process Path does not exist!");

        string path = Path.GetDirectoryName(processPath);
        if (path == null)
            throw new Exception("Process Directory does not exist!");

        int index = path.IndexOf("bin", StringComparison.Ordinal);

        return index < 0 ? path : path.Remove(index);
    }
}