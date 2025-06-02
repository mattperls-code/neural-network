# Neural Network

A flexible all purpose multilayer perceptron neural network class.

The `NeuralNetwork` class handles generating network parameters, feedforward evaluation, loss metrics, single or batch training with gradient descent, and stores input loss gradients for easy chaining.

```cpp
/* Create Neural Network Instance */

NeuralNetwork myNeuralNetwork(
    3, // input node count
    {
        HiddenLayerParameters(10, RELU), // hidden layer with 10 nodes and relu activation
        HiddenLayerParameters(16, TANH), // hidden layer with 16 nodes and tanh activation
        
        HiddenLayerParameters(4, LINEAR) // output layer with 4 nodes and linear activation
    },
    SOFTMAX, // normalization function
    CATEGORICAL_CROSS_ENTROPY // loss function
);
```

```cpp
/* Initialize Parameters */

// optionally set distribution range for initial weights and bias
myNeuralNetwork.initializeRandomLayerParameters();
```

```cpp
/* Use DataPoint Class For Training */

std::vector<DataPoint> trainingDataPoints;

trainingDataPoints.emplace_back(
    Matrix({{ 0.2 }, { -0.4 }, { 0.3 }}), // input
    Matrix({{ 0.01 }, { 0.97 }, { 0.01 }, { 0.01 }}) // expected output
);

// ... add other training data ...
```

```cpp
/* Train Using A Single DataPoint */

float learningRate = 0.05;

myNeuralNetwork.train(trainingDataPoints[0], learningRate);
```

```cpp
/* Train Using A Batch */

float learningRate = 0.1;

myNeuralNetwork.batchTrain(trainingDataPoints, learningRate);
```

```cpp
/* Make A Prediction */

Matrix input({{ 0.3 }, { 0.7 }, { -0.2 }});

Matrix output = myNeuralNetwork.calculateFeedForwardOutput(input);
```

```cpp
/* Calculate The Loss For A Known Point */

Matrix expectedOutput({{ 0.97 }, { 0.01 }, { 0.01 }, { 0.01 }});

float loss = myNeuralNetwork.calculateLoss(input, expectedOutput);
```

## Examples

### 2D Region Based Classification

![After 0 Training Cycles](./results/classification2D/regions/after0.png)
![After 1000 Training Cycles](./results/classification2D/regions/after1000.png)
![After 3000 Training Cycles](./results/classification2D/regions/after3000.png)
![Benchmark](./results/classification2D/regions/benchmark.png)

### Iris Species Classification

![Benchmark](./results/classification/iris/benchmark.png)

### Parametric Function Approximation

#### Circle

![After 0 Training Cycles](./results/nonlinearFitParametricCurve/circle/after0.png)
![After 1000 Training Cycles](./results/nonlinearFitParametricCurve/circle/after1000.png)
![After 5000 Training Cycles](./results/nonlinearFitParametricCurve/circle/after5000.png)

#### Star

![After 0 Training Cycles](./results/nonlinearFitParametricCurve/star/after0.png)
![After 1000 Training Cycles](./results/nonlinearFitParametricCurve/star/after1000.png)
![After 5000 Training Cycles](./results/nonlinearFitParametricCurve/star/after5000.png)

### Polynomial Regression

#### Sin(x)

![After 0 Training Cycles](./results/polynomialFitParametricCurve/sin(x)/after0.png)
![After 1000 Training Cycles](./results/polynomialFitParametricCurve/sin(x)/after1000.png)
![After 5000 Training Cycles](./results/polynomialFitParametricCurve/sin(x)/after5000.png)

#### Cos(x)

![After 0 Training Cycles](./results/polynomialFitParametricCurve/cos(x)/after0.png)
![After 1000 Training Cycles](./results/polynomialFitParametricCurve/cos(x)/after1000.png)
![After 5000 Training Cycles](./results/polynomialFitParametricCurve/cos(x)/after5000.png)