namespace NeuralNetwork.Core.System;

public class MyParallel {

    // My own implementation of Parallel.For, mostly to learn how to work with threads
    public static void For(int startInclusive, int endExclusive, int stepSize) {
        int processors = Environment.ProcessorCount;
        if (processors <= 0)
            throw new Exception("No processors found. (How?)");

        int index = startInclusive;

        Thread thread = new Thread((i) => {
            Console.WriteLine(i);
        });
        
        thread.Start();
    }
}