# Neurtal-Network-Overview
An open-source project I built to understand and share what I've learned about neural networks and deep learning. It includes different implementations, training-loss graphs, time and space complexity analysis, and explanations of the mathematics behind the network.

The goal is to explore both how neural networks learn and how different implementations affect their performance.

This current implementation is not well-structured for usage, but most of what you need to know is inside the `Program.cs` file. If you see this repository and know me in real life, ask me whatever you need and hopefully I'll remember to answer. If not... thanks for finding me?

- The SIMD implementations are the best I could think of and I honestly won't be bothered to increase it anymore.
- As for the GPU, I improved it quiet a bit. It runs really fast, which I am extremely proud of, as well as fairly accurate, though it did have issues so this current version isn't too reliable for usage (Not that you should use this as a library, but anyway). I used Reduction Kernels and more abstract Kernels to achieve things such as the Sum/Max and softmax. Not sure if they can be improved, but they are really cool as of now.

# Special Thanks
- ## 3Blue1Brown's [Neural Networks Lessons](https://www.3blue1brown.com/?topic=neural-networks)
- ## Primer's [Simulating a single brain cell](www.youtube.com/watch?v=GkiITbgu0V0)

# Math

This section explains the network's math model and how it learns through back propagation and gradient descent.

We'll treat vectors as column vectors for simplicity with matrices.

## Notations
### $l =$ Layer, $L =$ Last Layer, $N_L =$ The Neurons at layer $L$

### $ReLU(x) = \begin{cases} x, x > 0 \\ 0, x \leq 0 \end{cases}$

- ### Weights: $W^{[l]} \in \mathbb{R}^{N_l \times N_{l-1}}$
- ### Bias: $b^{[l]} \in \mathbb{R}^{N_l}$
- ### Values: $z^{[l]} \in \mathbb{R}^{N_l}$
- ### Activations: $a^{[l]} \in \mathbb{R}^{N_l}$
---
- ### Input: $a^{[0]} = x$
- ### Target: $y \in {0,1}^{N_L}$
- ### Prediction: $\hat{y}=a^{[L]}$
- ### Loss: $C$
---
- ### Error Signal: $\delta^{[l]}=\frac{\partial C}{\partial z^{[l]}}$
---
- ### Learning Rate: $\eta$
- ### Mini-batch size: $\beta$

## Forward Propagation

Forward propagation is the process of moving from the starting layer to the final layer in order to produce the prediction.

### 1. Linear Transformation

Each layer takes the previous layer's activation, multiplies it by its weight matrix and then its bias.

$$
z^{[l]}=W^{[l]}a^{[l-1]}+b^{[l]}
$$

### 2. Activation functions

Activation functions filter the value in such a way that makes this layer's evaluations more helpful for the next layer.

For hidden layers, this network uses the *Rectified Linear Unit* (ReLU):

$$
\operatorname{ReLU}(x) = \max(x, 0)
$$
Therefore,
$$
a^{[l]}=\operatorname{ReLU}(z^{[l]}),
\qquad 1\leq l<L
$$

The output layer uses Softmax to convert its pre-activations into probabilities:

$$
\hat{y}_k=a_k^{[L]}
=\frac{e^{z_k^{[L]}}}
{\sum_{j=1}^{n_L}e^{z_j^{[L]}}}
$$

Softmax converts the last layer's values into probabilities, or the "confidence" value, of each guess. 1 being 100% sure and 0, 0%. The sum is also equal to 1.

## Loss Functions

To measure how well the network did, we use categorial cross-entropy:

$$
C=-\sum_{k=1}^{n_L}y_k\log(\hat{y}_k)
$$

Because the target is one-hot encoded, only the term corresponding to the correct answer contributes to the sum, which simplifies it to:

$$
C = -\log(\hat{y}_c)
$$

The more confident the network is on the correct answer, the lower this value will be, whereas less confidence will produce a higher loss.

## Back Propagation

Back propagation is similar to Forward propagation, just... backwards. Shocker.

In Forward propagation we used our input to compute a prediction. Now, we'll use that prediction along with our target value in order to see how we need to adjust the network in order to ensure it produces smarter results.

Spoiler warning, the process will look a lot like the chain rule.

### 1. Outer Layer

For a softmax output layer combined with cross-entropy loss, the gradient with respect to the previous layer's activation turns out to be
$$
\delta^{[L]}
=\frac{\partial C}{\partial z^{[L]}}
=\hat{y}-y
$$

### 2. Hidden Layers

For a hidden layer, the error signal is calculated by the next layer's error signal:

$$
\delta^{[l]}
=
\left(W^{[l+1]}\right)^T
\delta^{[l+1]}
\odot
\operatorname{ReLU}'(z^{[l]})
$$

for $l=L-1,L-2,\ldots,1$

Here, $\odot$ is element-wise multiplication, so $y[i] = x_0[i] \cdot x_1[i]$, and the derivative of ReLU is:

$$
\operatorname{ReLU}'(x)=
\begin{cases}
1 & x>0\\
0 & x\leq 0
\end{cases}
$$

The transpose of the weight appears because we are propagating the gradient backwards through the linear transformation.

More generally, the gradient with respend to the previous layer's activation is:
$$
\frac{\partial C}{\partial a^{[l-1]}}
=
\left(W^{[l]}\right)^T\delta^{[l]}
$$

And to explain further, starting from
$$
\frac{\partial C}{\partial W^{[l]}}
=
\frac{\partial C}{\partial a^{[l]}}
\cdot
\frac{\partial a^{[l]}}{\partial z^{[l]}}
\cdot
\frac{\partial z^{[l]}}{\partial W^{[l]}}
$$

- $$
\frac{\partial C}{\partial a^{[l]}} =
\left(W^{[l]}\right)^T\delta^{[l]}
$$
- $$
\frac{\partial a^{[l]}}{\partial z^{[l]}} =
\operatorname{ReLU}'(z^{[l]})
$$

Which again, gives us:
$$
\delta^{[l]}
=
\left(W^{[l+1]}\right)^T
\delta^{[l+1]}
\odot
\operatorname{ReLU}'(z^{[l]})
$$
and leaving us with
$$
\frac{\partial z^{[l]}}{\partial W^{[l]}} = 
a^{[l-1]}
$$
$$
\frac{\partial z^{[l]}}{\partial b^{[l]}} = 1
$$
$$
\frac{\partial z^{[l]}}{\partial a^{[l - 1]}} = 
W^{[l]}
$$
for the weights, biases and activations respectively.

We already explained the last one, so the first two that are left as simply the changes to the weights and bias for that layer (multipled by $\delta$, of course)

## Adjusting

After all that, we finally know how much we need to change our parameters.
Changing them is as simply as a quick subtraction.

$$
W^{[L]} -=
(\delta^{[l]} \cdot a^{[l-1]})
\cdot
\frac{\eta}{\beta}
$$
$$
b^{[L]} -=
\delta^{[l]}
\cdot
\frac{\eta}{\beta}
$$

Where $\eta$ is our learning rate and $\beta$ our batch size, which speaking of...

### Learning Process

As you can tell, this process is more of an approximation than anything else. A "simple" function that predicts something, given data. However, how do we train it?

We give it data, commonly referred as *Training Data*, which is an array of inputs, with an index, or array of values (depending on the network). For our example, we have a simple index which contains the digit the image represents.

Before I continue, I want you to image something. Had I told the network to simply add the gradient directly to the weights and biases, it'd be like telling it to take anything it sees literally and changing its whole reality around that single example, since it'd take a full step towards that example. Well, this is why the learning rate exists. It's a variable that tells the network how big steps it should take each time.
The learning rate can either be static or dynamic, slowly decaying to ensure the network doesn't overshoot answers.

That aside though, here's another issue. Training one example at a time can be slow and sometimes inaccurate, so her introduce a *Data Batch* that has a subset of our total Training Data and trains it together. In short, it adds up all the example's opinions on how the network should move before adding the average back to the network.

This is how we end up with th scala.
## $\frac{\eta}{\beta}$

A few more things regarding out Training Data.

For starters, training data can be time-consuming to create, so it's reasonable to recycle our already acquired data. However, if we give the network the same patterns, it'll eventually cheat it's way to victory, so to avoid that, we make sure to shuffle the training data before beginning the training process.

As for the testing, we need to make sure the model/network can hold well on its own with unknown examples, so we take a small portion of our data and keep it for testing, so once the model has trained enough, we give it this new, unknown data and have a good approximation of it's true potential.

---

That's all I had to say. I am not that good with writing, so if anything is difficult to keep up with, please let me know and I'll do my best to improve it.

Thank you for reading!