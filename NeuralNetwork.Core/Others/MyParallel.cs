namespace NeuralNetwork.Core.Others;

public static class MyParallel {
    /**
     * My own implementation of Parallel.For. Main difference, the finalize function starts once all the work finished,
     * whereas Parallel.For starts the finalize function right as the thread finishes, making it undetermined.
     * <param name="startInclusive">
     * The beginning of the for loop.
     * </param>
     * <param name="endExclusive">
     * The exclusive end of the for loop.
     * </param>
     * <param name="createLocal">
     * The initializing function for each worker.
     * </param>
     * <param name="process">
     * The processing function each worker will have to do.
     * </param>
     * <param name="finalize">
     * The finalizing function for each worker, after all work is completed.
     * </param>
     */
    public static void For<TLocal>(int startInclusive, int endExclusive,
        Func<TLocal> createLocal, Action<int, TLocal> process, Action<TLocal> finalize) {
        For(startInclusive, endExclusive, Environment.ProcessorCount, createLocal, process, finalize);
    }

    /**
     * My own implementation of Parallel.For. Main difference, the finalize function starts once all the work finished,
     * whereas Parallel.For starts the finalize function right as the thread finishes, making it undetermined.
     * <param name="startInclusive">
     * The beginning of the for loop.
     * </param>
     * <param name="endExclusive">
     * The exclusive end of the for loop.
     * </param>
     * <param name="processors">
     * The amount of processors that will be used. Will throw an error if the system doesn't support that amount.
     * </param>
     * <param name="createLocal">
     * The initializing function for each worker.
     * </param>
     * <param name="process">
     * The processing function each worker will have to do.
     * </param>
     * <param name="finalize">
     * The finalizing function for each worker, after all work is completed.
     * </param>
     */
    public static void For<TLocal>(int startInclusive, int endExclusive, int processors,
        Func<TLocal> createLocal, Action<int, TLocal> process, Action<TLocal> finalize) {
        /*
         * Not sure how they could be lower than 0, but just to be safe.
         */
        if (processors <= 0)
            throw new Exception("No processors found. (How?)");
        
        /*
         * Validation of the total steps
         */
        int totalSteps = endExclusive - startInclusive;
        if (totalSteps <= 0) return;
        
        /*
         * Cap the processors to the total steps, after all, no need to use 16 processors for an operator of 10. for example.
         */
        processors = Math.Min(processors, totalSteps);
        
        /*
         * Initialize all processors to the createLocal function
         */
        TLocal[] workers = new TLocal[processors];
        for (int i = 0; i < workers.Length; i++) {
            workers[i] = createLocal();
        }
        
        /*
         * Creates the threads that are going to be used and gives them their work to do, split almost equally.
         */
        Thread[] threads = new Thread[processors];
        for (int w = 0; w < workers.Length; w++) {
            threads[w] = new Thread(rank => {
                int id = (int)rank!;
                
                int start = startInclusive + (endExclusive - startInclusive) * id / processors;
                int finish = startInclusive + (endExclusive - startInclusive) * (id + 1) / processors;
                for (int i = start; i < finish; i++) {
                    process(i, workers[id]);
                }
            });
            threads[w].Start(w);
        }

        /*
         * Waits for all the threads to finish their work before continuing.
         */
        foreach (Thread thread in threads) {
            thread.Join();
        }
        
        /*
         * Finalizes each worker.
         */
        for (int w = 0; w < workers.Length; w++) {
            threads[w] = new Thread(rank => {
                int id = (int)rank!;
                
                finalize(workers[id]);
            });
            threads[w].Start(w);
        }

        /*
         * Waits for them to finish...
         */
        foreach (Thread thread in threads) {
            thread.Join();
        }
    }
}