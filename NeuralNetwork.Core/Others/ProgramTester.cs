using System.Text;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using NeuralNetwork.Core.Backend;
using NeuralNetwork.Core.Data;
using NeuralNetwork.Core.Data.SingleValues;
using NeuralNetwork.Core.Data.Tensors;
using NeuralNetwork.Core.NetworkFormats.ExecutableNetworks;
using NeuralNetwork.Core.NetworkFormats.Layers;
using NeuralNetwork.Core.NetworkFormats.Layers.State;
using NeuralNetwork.Core.NetworkFormats.Networks;

namespace NeuralNetwork.Core.Others;

/* Code made by ChatGPT
 * Didn't want to bother xd
 */

public static class ProgramTester {
	private const float Tolerance = 0.0005f;

	public static void CheckProgram(bool debug = false) {
		using Context context = Context.Create(builder => builder.Cuda().EnableAlgorithms());

		Device device = context.Devices.FirstOrDefault()
		                ?? throw new InvalidOperationException("No CUDA device was found. The tester requires an NVIDIA CUDA device.");
		using Accelerator accelerator = device.CreateAccelerator(context);

		ScalarBackend scalar = new ScalarBackend();
		SimdBackend simd = new SimdBackend();
		IlgpuBackend ilGpu = new IlgpuBackend(accelerator);

		IBackend[] backends = [scalar, simd, ilGpu];
		int mistakes = 0;

		mistakes += RunCheck("Activation/loss initialization", () => CheckAcivationInitializing(backends, debug), debug);
		mistakes += RunCheck("Dropout behavior", () => CheckDropoutRandomness(backends, debug), debug);
		mistakes += RunCheck("Activation and loss operations", () => CheckActivationAndLossOperations(backends, debug), debug);
		mistakes += RunCheck("Tensor/layer shape validation", () => CheckShapeValidation(backends, debug), debug);
		mistakes += RunCheck("Fully connected forward/backward", () => CheckFullyConnectedForwardBackward(backends, debug), debug);
		mistakes += RunCheck("VirtualNetwork gradients", () => CheckVirtualNetworkGradients(backends, debug), debug);

		Console.WriteLine($"Program Tester completed with {mistakes} mistakes");
	}

	private static int RunCheck(string name, Func<int> check, bool debug) {
		try {
			int mistakes = check();
			if (debug || mistakes > 0) {
				Console.WriteLine($"{name}: {(mistakes == 0 ? "PASSED" : $"FAILED ({mistakes})")}");
				Console.WriteLine();
			}
			return mistakes;
		} catch (Exception exception) {
			Console.WriteLine($"{name}: CRASHED ({exception.GetType().Name}: {exception.Message})");
			if (debug) Console.WriteLine(exception);
			return 1;
		}
	}

	private static int CheckAcivationInitializing(IBackend[] backends, bool debug) {
		int mistakes = 0;

		ActivationFunction activation = new ReLu();
		activation.Initialize(backends[0]);

		bool activationRejected = false;
		try {
			activation.Initialize(backends[1]);
		} catch (InvalidOperationException) {
			activationRejected = true;
		}

		if (!activationRejected) {
			mistakes++;
			if (debug) Console.WriteLine("Activation re-initialization was not rejected.");
		}

		LossFunction loss = new CrossEntropy();
		loss.Initialize(backends[0]);

		bool lossRejected = false;
		try {
			loss.Initialize(backends[1]);
		} catch (InvalidOperationException) {
			lossRejected = true;
		}

		if (!lossRejected) {
			mistakes++;
			if (debug) Console.WriteLine("Loss re-initialization was not rejected.");
		}

		return mistakes;
	}

	private static int CheckDropoutRandomness(IBackend[] backends, bool debug) {
		const float chance = 0.25f;
		const int size = 1024;
		const int loops = 100;
		const float allowedDeviation = 0.01f;

		int mistakes = 0;
		float invertedDropoutScale = 1f / (1f - chance);

		foreach (IBackend backend in backends) {
			string backendName = backend.GetType().Name;
			using DropoutLayer layer = new DropoutLayer(chance, 100);
			layer.Initialize(size, backend, null);

			using ILayerForwardState stateA = layer.CreateForwardState();
			using ILayerForwardState stateB = layer.CreateForwardState();
			DropoutForwardState dropoutA = (DropoutForwardState)stateA;

			float[] ones = new float[size];
			Array.Fill(ones, 1f);
			using ITensor<float> input = backend.CreateTensor(ones, size);
			using ITensor<float> deltaOut = backend.CreateTensor(0f, size);

			layer.Forward(input, stateA, null, PassMode.Training);
			float[] firstMaskOutput = backend.Read(stateA.Output);

			layer.Forward(input, stateB, null, PassMode.Training);
			float[] secondMaskOutput = backend.Read(stateB.Output);

			if (firstMaskOutput.SequenceEqual(secondMaskOutput)) {
				mistakes++;
				if (debug) Console.WriteLine($"{backendName}: two dropout states generated identical first masks.");
			}

			int dropped = 0;
			int invalidValues = 0;
			float[] previousOutput = firstMaskOutput;

			for (int loop = 0; loop < loops; loop++) {
				float[] currentOutput;
				if (loop == 0) {
					currentOutput = firstMaskOutput;
				} else {
					layer.Forward(input, stateA, null, PassMode.Training);
					currentOutput = backend.Read(stateA.Output);
				}

				foreach (float value in currentOutput) {
					if (value == 0f) dropped++;
					else if (Math.Abs(value - invertedDropoutScale) > Tolerance) invalidValues++;
				}

				if (loop > 0 && currentOutput.SequenceEqual(previousOutput)) {
					mistakes++;
					if (debug) Console.WriteLine($"{backendName}: consecutive dropout masks were identical.");
				}

				previousOutput = currentOutput;
			}

			float observedChance = dropped / (float)(size * loops);
			if (Math.Abs(observedChance - chance) > allowedDeviation) {
				mistakes++;
				if (debug)
					Console.WriteLine($"{backendName}: dropout rate was {observedChance:P2}; expected {chance:P2}.");
			}

			if (invalidValues > 0) {
				mistakes++;
				if (debug) Console.WriteLine($"{backendName}: {invalidValues} dropout outputs had an unexpected value.");
			}

			// The backward pass must apply the exact mask generated by the forward pass.
			ILayerBackwardState backwardState = layer.CreateBackwardState();
			
			backend.Set(backwardState.Delta, ones);
			layer.Delta(dropoutA, backwardState, deltaOut);
			float[] backward = backend.Read(deltaOut);
			float[] mask = backend.Read(dropoutA.Mask);
			if (!NearlyEqual(backward, mask)) {
				mistakes++;
				if (debug) Console.WriteLine($"{backendName}: dropout backward pass did not apply the forward mask.");
			}

			// Inference must not apply dropout or rescale the input.
			float[] inferenceValues = Enumerable.Range(0, size).Select(i => i * 0.125f - 3f).ToArray();
			using ITensor<float> inferenceInput = backend.CreateTensor(inferenceValues, size);
			layer.Forward(inferenceInput, stateA, null, PassMode.Inference);
			if (!NearlyEqual(backend.Read(stateA.Output), inferenceValues)) {
				mistakes++;
				if (debug) Console.WriteLine($"{backendName}: dropout modified values during inference.");
			}
		}

		return mistakes;
	}

	private static int CheckActivationAndLossOperations(IBackend[] backends, bool debug) {
		int mistakes = 0;

		foreach (IBackend backend in backends) {
			string backendName = backend.GetType().Name;

			// Leaky ReLU forward and backward.
			float[] values = [-2f, -1f, 1f, 2f];
			using ITensor<float> input = backend.CreateTensor(values, values.Length);
			using ITensor<float> output = backend.CreateTensor(0f, values.Length);
			using ITensor<float> gradient = backend.CreateTensor(1f, values.Length);

			ActivationFunction leakyRelu = new LeakyReLu(0.1f);
			leakyRelu.Initialize(backend);
			leakyRelu.Forward(input, output, null);
			if (!NearlyEqual(backend.Read(output), [-0.2f, -0.1f, 1f, 2f])) {
				mistakes++;
				if (debug) Console.WriteLine($"{backendName}: LeakyReLU forward result was incorrect.");
			}

			leakyRelu.Backward(input, gradient);
			if (!NearlyEqual(backend.Read(gradient), [0.1f, 0.1f, 1f, 1f])) {
				mistakes++;
				if (debug) Console.WriteLine($"{backendName}: LeakyReLU backward result was incorrect.");
			}

			// Sigmoid forward and backward.
			backend.Set(gradient, [1f, 1f, 1f, 1f]);
			ActivationFunction sigmoid = new Sigmoid();
			sigmoid.Initialize(backend);
			sigmoid.Forward(input, output, null);

			float[] sigmoidExpected = values.Select(v => 1f / (1f + MathF.Exp(-v))).ToArray();
			if (!NearlyEqual(backend.Read(output), sigmoidExpected)) {
				mistakes++;
				if (debug) Console.WriteLine($"{backendName}: Sigmoid forward result was incorrect.");
			}

			sigmoid.Backward(input, gradient);
			float[] sigmoidGradientExpected = sigmoidExpected.Select(v => v * (1f - v)).ToArray();
			if (!NearlyEqual(backend.Read(gradient), sigmoidGradientExpected)) {
				mistakes++;
				if (debug) Console.WriteLine($"{backendName}: Sigmoid backward result was incorrect.");
			}

			// Softmax forward and its Jacobian-vector product.
			float[] logits = [0f, 1f, -1f];
			using ITensor<float> softmaxInput = backend.CreateTensor(logits, logits.Length);
			using ITensor<float> softmaxOutput = backend.CreateTensor(0f, logits.Length);
			float[] upstream = [1f, 2f, 3f];
			using ITensor<float> softmaxGradient = backend.CreateTensor(upstream, logits.Length);
			using ISingleValue<int> prediction = backend.CreateValue(-1);

			ActivationFunction softmax = new Softmax();
			softmax.Initialize(backend);
			softmax.Forward(softmaxInput, softmaxOutput, prediction);

			float[] probabilities = SoftmaxValues(logits);
			if (!NearlyEqual(backend.Read(softmaxOutput), probabilities) || backend.Read(prediction) != 1) {
				mistakes++;
				if (debug) Console.WriteLine($"{backendName}: Softmax forward result or prediction was incorrect.");
			}

			float dot = probabilities.Zip(upstream, (p, g) => p * g).Sum();
			float[] softmaxGradientExpected = probabilities.Zip(upstream, (p, g) => p * (g - dot)).ToArray();
			softmax.Backward(softmaxInput, softmaxGradient);
			if (!NearlyEqual(backend.Read(softmaxGradient), softmaxGradientExpected)) {
				mistakes++;
				if (debug) Console.WriteLine($"{backendName}: Softmax backward result was incorrect.");
			}

			// Loss calculation and output gradients.
			float[] activationValues = [0.1f, 0.6f, 0.3f];
			float[] expectedValues = [0f, 1f, 0f];
			using ITensor<float> activations = backend.CreateTensor(activationValues, activationValues.Length);
			using ITensor<float> expected = backend.CreateTensor(expectedValues, expectedValues.Length);
			using ITensor<float> lossGradient = backend.CreateTensor(0f, activationValues.Length);
			using ISingleValue<int> label = backend.CreateValue(1);
			using ISingleValue<float> lossValue = backend.CreateValue(0f);

			ILossBackend lossBackend = (ILossBackend)backend;
			lossBackend.MSE_Calculate(activations, label, 1f, lossValue);
			if (Math.Abs(backend.Read(lossValue) - 0.26f) > Tolerance) {
				mistakes++;
				if (debug) Console.WriteLine($"{backendName}: MSE calculation was incorrect.");
			}

			lossBackend.MSE_Backward(activations, expected, 1f, lossGradient);
			if (!NearlyEqual(backend.Read(lossGradient), [0.2f, -0.8f, 0.6f])) {
				mistakes++;
				if (debug) Console.WriteLine($"{backendName}: MSE backward result was incorrect.");
			}

			lossBackend.CrossEntropy_Calculate(activations, label, lossValue);
			float expectedCrossEntropy = -MathF.Log(activationValues[1] + 1e-15f);
			if (Math.Abs(backend.Read(lossValue) - expectedCrossEntropy) > Tolerance) {
				mistakes++;
				if (debug) Console.WriteLine($"{backendName}: CrossEntropy calculation was incorrect.");
			}

			lossBackend.CrossEntropy_Backward(activations, expected, lossGradient);
			if (!NearlyEqual(backend.Read(lossGradient), [0.1f, -0.4f, 0.3f])) {
				mistakes++;
				if (debug) Console.WriteLine($"{backendName}: CrossEntropy backward result was incorrect.");
			}
		}

		return mistakes;
	}

	private static int CheckShapeValidation(IBackend[] backends, bool debug) {
		int mistakes = 0;

		foreach (IBackend backend in backends) {
			string backendName = backend.GetType().Name;

			bool invalidTensorShapeRejected = false;
			try {
				using ITensor<float> invalid = backend.CreateTensor(0f, 0);
			} catch (ArgumentException) {
				invalidTensorShapeRejected = true;
			}

			if (!invalidTensorShapeRejected) {
				mistakes++;
				if (debug) Console.WriteLine($"{backendName}: accepted a tensor dimension of zero.");
			}

			bool invalidLayerDataRejected = false;
			using (FullyConnectedLayer invalidLayer = new FullyConnectedLayer(2, new ReLu())) {
				try {
					invalidLayer.InitializeFromData(3, backend, [new float[5], new float[2]]);
				} catch (ArgumentException) {
					invalidLayerDataRejected = true;
				}
			}

			if (!invalidLayerDataRejected) {
				mistakes++;
				if (debug) Console.WriteLine($"{backendName}: accepted fully connected weights with an invalid shape.");
			}
		}

		return mistakes;
	}

	private static int CheckFullyConnectedForwardBackward(IBackend[] backends, bool debug) {
		int mistakes = 0;

		float[] initialWeights = [1f, 2f, -1f, -1f, 0.5f, 3f];
		float[] initialBias = [0.5f, -0.5f];
		float[] inputValues = [1f, 2f, -1f];

		foreach (IBackend backend in backends) {
			string backendName = backend.GetType().Name;
			using FullyConnectedLayer layer = new FullyConnectedLayer(2, new ReLu());
			layer.InitializeFromData(3, backend, [initialWeights, initialBias]);

			using ILayerForwardState forwardState = layer.CreateForwardState();
			using FullyConnectedBackwardState backwardState = (FullyConnectedBackwardState)layer.CreateBackwardState();
			using ITensor<float> input = backend.CreateTensor(inputValues, inputValues.Length);
			using ITensor<float> inputGradient = backend.CreateTensor(0f, inputValues.Length);

			layer.Forward(input, forwardState, null, PassMode.Training);
			if (!NearlyEqual(backend.Read(forwardState.Output), [6.5f, 0f])) {
				mistakes++;
				if (debug) Console.WriteLine($"{backendName}: fully connected forward pass was incorrect.");
			}

			backend.Set(backwardState.Delta, [2f, 3f]);
			layer.ActivationDelta(forwardState, backwardState);
			if (!NearlyEqual(backend.Read(backwardState.Delta), [2f, 0f])) {
				mistakes++;
				if (debug) Console.WriteLine($"{backendName}: fully connected activation derivative was incorrect.");
			}

			layer.BackPropagation(input, backwardState);
			if (!NearlyEqual(backend.Read(backwardState.WeightsDelta), [2f, 4f, -2f, 0f, 0f, 0f]) ||
			    !NearlyEqual(backend.Read(backwardState.BiasDelta), [2f, 0f])) {
				mistakes++;
				if (debug) Console.WriteLine($"{backendName}: fully connected parameter gradients were incorrect.");
			}

			layer.Delta(forwardState, backwardState, inputGradient);
			if (!NearlyEqual(backend.Read(inputGradient), [2f, 4f, -2f])) {
				mistakes++;
				Console.WriteLine(string.Join(", ", backend.Read(inputGradient)));
				if (debug) Console.WriteLine($"{backendName}: fully connected input gradient was incorrect.");
			}

			layer.Adjust(backwardState, 0.1f);
			float[] expectedWeightsAfterAdjust = [0.8f, 1.6f, -0.8f, -1f, 0.5f, 3f];
			float[] actualWeightsAfterAdjust = layer.GetWeights().SelectMany(row => row).ToArray();
			if (!NearlyEqual(actualWeightsAfterAdjust, expectedWeightsAfterAdjust)) {
				mistakes++;
				if (debug) Console.WriteLine($"{backendName}: weight adjustment was incorrect.");
			}

			// A zero input isolates the bias, verifying that it was adjusted too.
			backend.Set(input, [0f, 0f, 0f]);
			layer.Forward(input, forwardState, null, PassMode.Inference);
			if (!NearlyEqual(backend.Read(forwardState.Output), [0.3f, 0f])) {
				mistakes++;
				if (debug) Console.WriteLine($"{backendName}: bias adjustment was incorrect.");
			}
		}

		return mistakes;
	}

	private static int CheckVirtualNetworkGradients(IBackend[] backends, bool debug) {
		int mistakes = 0;

		foreach (IBackend backend in backends) {
			mistakes += CheckVirtualNetworkGradient(
				backend,
				new MeanSquaredError(),
				new Sigmoid(),
				outputSize: 1,
				label: 0,
				useSoftmaxCrossEntropy: false,
				debug: debug
			);

			mistakes += CheckVirtualNetworkGradient(
				backend,
				new CrossEntropy(),
				new Softmax(),
				outputSize: 3,
				label: 1,
				useSoftmaxCrossEntropy: true,
				debug: debug
			);
		}

		return mistakes;
	}

	private static int CheckVirtualNetworkGradient(
		IBackend backend,
		LossFunction loss,
		ActivationFunction activation,
		int outputSize,
		int label,
		bool useSoftmaxCrossEntropy,
		bool debug
	) {
		string backendName = backend.GetType().Name;
		const float rawInput = 0.5f;
		const float learningScale = 0.05f;

		using Network network = new Network(
			backend,
			loss,
			new Random(12345),
			new InputLayer(1),
			new FullyConnectedLayer(outputSize, activation)
		);

		NetworkParameters before = ReadSingleFullyConnectedParameters(network);
		float normalizedInput = (rawInput - 0.1307f) / 0.3081f;

		float[] logits = new float[outputSize];
		for (int i = 0; i < outputSize; i++)
			logits[i] = before.Weights[i] * normalizedInput + before.Bias[i];

		float[] outputValues = useSoftmaxCrossEntropy
			? SoftmaxValues(logits)
			: logits.Select(value => 1f / (1f + MathF.Exp(-value))).ToArray();

		float[] expectedDelta = new float[outputSize];
		for (int i = 0; i < outputSize; i++) {
			float expected = i == label ? 1f : 0f;
			if (useSoftmaxCrossEntropy) {
				// The CrossEntropy + Softmax combination returns dL/d(logits) directly.
				expectedDelta[i] = outputValues[i] - expected;
			} else {
				// MSE returns dL/d(activation); Sigmoid's derivative must still be applied.
				float dLossDActivation = 2f * (outputValues[i] - expected);
				float sigmoidDerivative = outputValues[i] * (1f - outputValues[i]);
				expectedDelta[i] = dLossDActivation * sigmoidDerivative;
			}
		}

		float[] expectedWeights = new float[outputSize];
		float[] expectedBias = new float[outputSize];
		for (int i = 0; i < outputSize; i++) {
			expectedWeights[i] = before.Weights[i] - learningScale * expectedDelta[i] * normalizedInput;
			expectedBias[i] = before.Bias[i] - learningScale * expectedDelta[i];
		}

		using TrainableNetwork trainableNetwork = new TrainableNetwork(network, 1);
		DataSample<float> sample = new DataSample<float>([rawInput], label);
		trainableNetwork.LoadBuffer(new DataBatch<float>(sample));
		trainableNetwork.ProcessFullBuffer(() => {
			trainableNetwork.CompleteForwardPass(network);
			trainableNetwork.IterateBackward(network);
		});

		int expectedPrediction = Array.IndexOf(logits, logits.Max());
		if (trainableNetwork.RawGuess(0).Prediction != expectedPrediction) {
			if (debug) Console.WriteLine($"{backendName}: VirtualNetwork produced an unexpected prediction ({(useSoftmaxCrossEntropy ? "Softmax/CE" : "Sigmoid/MSE")}).");
			return 1;
		}

		trainableNetwork.IterateAdjust(network, learningScale);
		NetworkParameters after = ReadSingleFullyConnectedParameters(network);

		int mistakes = 0;
		if (!NearlyEqual(after.Weights, expectedWeights) || !NearlyEqual(after.Bias, expectedBias)) {
			mistakes++;
			if (debug) {
				string combination = useSoftmaxCrossEntropy ? "Softmax + CrossEntropy" : "Sigmoid + MSE";
				Console.WriteLine($"{backendName}: VirtualNetwork parameter update was incorrect for {combination}.");
				Console.WriteLine($"  Expected weights: [{string.Join(", ", expectedWeights)}]");
				Console.WriteLine($"  Actual weights:   [{string.Join(", ", after.Weights)}]");
				Console.WriteLine($"  Expected bias:    [{string.Join(", ", expectedBias)}]");
				Console.WriteLine($"  Actual bias:      [{string.Join(", ", after.Bias)}]");
			}
		}

		return mistakes;
	}

	private sealed record NetworkParameters(float[] Weights, float[] Bias);

	/**
	 * Reads the parameters from a network containing exactly an InputLayer and one FullyConnectedLayer.
	 * This uses the existing serialization format so the tester does not need public bias accessors.
	 */
	private static NetworkParameters ReadSingleFullyConnectedParameters(Network network) {
		using MemoryStream stream = new MemoryStream();
		using (BinaryWriter writer = new BinaryWriter(stream, Encoding.UTF8, leaveOpen: true))
			network.Save(writer);

		stream.Position = 0;
		using BinaryReader reader = new BinaryReader(stream, Encoding.UTF8, leaveOpen: true);

		int layerAmount = reader.ReadInt32();
		if (layerAmount != 2)
			throw new InvalidDataException($"Expected a two-layer network, got {layerAmount} layers.");

		if (reader.ReadString() != "input")
			throw new InvalidDataException("Expected the first layer to be an InputLayer.");
		reader.ReadInt32(); // Input size.

		if (reader.ReadString() != "fully_connected")
			throw new InvalidDataException("Expected the second layer to be FullyConnectedLayer.");

		int outputSize = reader.ReadInt32();
		int inputSize = reader.ReadInt32();
		reader.ReadString(); // Activation identifier.

		float[] weights = new float[checked(outputSize * inputSize)];
		float[] bias = new float[outputSize];

		for (int i = 0; i < weights.Length; i++) weights[i] = reader.ReadSingle();
		for (int i = 0; i < bias.Length; i++) bias[i] = reader.ReadSingle();

		if (inputSize != 1)
			throw new InvalidDataException($"Expected one input neuron, got {inputSize}.");

		return new NetworkParameters(weights, bias);
	}

	private static float[] SoftmaxValues(float[] logits) {
		float max = logits.Max();
		float[] values = logits.Select(value => MathF.Exp(value - max)).ToArray();
		float sum = values.Sum();
		for (int i = 0; i < values.Length; i++) values[i] /= sum;
		return values;
	}

	private static bool NearlyEqual(float[] actual, float[] expected, float tolerance = Tolerance) {
		if (actual.Length != expected.Length) return false;
		for (int i = 0; i < actual.Length; i++) {
			if (!float.IsFinite(actual[i]) || Math.Abs(actual[i] - expected[i]) > tolerance)
				return false;
		}
		return true;
	}
}
