#include "neural_network.hpp"

// unary activation functions

Matrix UnaryActivationFunctionImplementation::evaluateLinear(const Matrix& values)
{
    return values;
}

Matrix UnaryActivationFunctionImplementation::linearDerivative(const Matrix& values, const Matrix&)
{
    return Matrix(values.shape(), 1.0);
}

Matrix UnaryActivationFunctionImplementation::evaluateRelu(const Matrix& values)
{
    Matrix output = values;

    for (auto& value : output.dangerouslyGetData()) if (value < 0.0) value = 0.0;

    return output;
}

Matrix UnaryActivationFunctionImplementation::reluDerivative(const Matrix& values, const Matrix&)
{
    Matrix output = values;

    for (auto& value : output.dangerouslyGetData()) value = value > 0.0 ? 1.0 : 0.0;

    return output;
}

Matrix UnaryActivationFunctionImplementation::evaluateSigmoid(const Matrix& values)
{
    Matrix output = values;

    for (auto& value : output.dangerouslyGetData()) value = 1.0 / (1.0 + exp(-value));

    return output;
}

Matrix UnaryActivationFunctionImplementation::sigmoidDerivative(const Matrix&, const Matrix& activatedValues)
{
    Matrix output = activatedValues;

    for (auto& value : output.dangerouslyGetData()) value *= 1.0 - value;

    return output;
}

Matrix UnaryActivationFunctionImplementation::evaluateTanh(const Matrix& values)
{
    Matrix output = values;

    for (auto& value : output.dangerouslyGetData()) value = tanh(value);

    return output;
}

Matrix UnaryActivationFunctionImplementation::tanhDerivative(const Matrix&, const Matrix& activatedValues)
{
    Matrix output = activatedValues;

    for (auto& value : output.dangerouslyGetData()) value = 1.0 - value * value;

    return output;
}

Matrix evaluateUnaryActivationFunction(UnaryActivationFunction unaryActivationFunction, const Matrix& values)
{
    if (unaryActivationFunction == LINEAR) return UnaryActivationFunctionImplementation::evaluateLinear(values);
    if (unaryActivationFunction == RELU) return UnaryActivationFunctionImplementation::evaluateRelu(values);
    if (unaryActivationFunction == SIGMOID) return UnaryActivationFunctionImplementation::evaluateSigmoid(values);
    if (unaryActivationFunction == TANH) return UnaryActivationFunctionImplementation::evaluateTanh(values);

    throw std::runtime_error("evaluateUnaryActivationFunction: unhandled unaryActivationFunction");
};

Matrix unaryActivationFunctionDerivative(UnaryActivationFunction unaryActivationFunction, const Matrix& values, const Matrix& activatedValues)
{
    if (unaryActivationFunction == LINEAR) return UnaryActivationFunctionImplementation::linearDerivative(values, activatedValues);
    if (unaryActivationFunction == RELU) return UnaryActivationFunctionImplementation::reluDerivative(values, activatedValues);
    if (unaryActivationFunction == SIGMOID) return UnaryActivationFunctionImplementation::sigmoidDerivative(values, activatedValues);
    if (unaryActivationFunction == TANH) return UnaryActivationFunctionImplementation::tanhDerivative(values, activatedValues);

    throw std::runtime_error("unaryActivationFunctionDerivative: unhandled unaryActivationFunction");
};

// normalization functions

Matrix NormalizationFunctionImplementation::evaluateIdentity(const Matrix& values)
{
    return values;
}

Matrix NormalizationFunctionImplementation::identityDerivative(const Matrix& values, const Matrix&)
{
    return Matrix(values.shape(), 1.0);
}

Matrix NormalizationFunctionImplementation::evaluateSoftmax(const Matrix& values)
{
    Matrix output = values;

    auto max = output.get(0, 0);
    for (auto value : output.dangerouslyGetData()) if (value > max) max = value;

    auto sum = 0.0;
    for (auto& value : output.dangerouslyGetData()) {
        value = exp(value - max);
        sum += value;
    }

    if (sum != 0) for (auto& value : output.dangerouslyGetData()) value /= sum;

    return output;
}

Matrix NormalizationFunctionImplementation::softmaxDerivative(const Matrix& values, const Matrix& normalizedValues)
{
    Matrix output(Shape(values.rowCount(), values.rowCount()));

    for (int i = 0;i<normalizedValues.rowCount();i++) {
        for (int j = 0;j<normalizedValues.rowCount();j++) {
            auto value = (i == j) ? (normalizedValues.get(i, 0) * (1.0 - normalizedValues.get(i, 0))) : (-normalizedValues.get(i, 0) * normalizedValues.get(j, 0));

            output.set(i, j, value);
        }
    }

    return output;
}

Matrix evaluateNormalizationFunction(NormalizationFunction normalizationFunction, const Matrix& values)
{
    if (normalizationFunction == IDENTITY) return NormalizationFunctionImplementation::evaluateIdentity(values);
    if (normalizationFunction == SOFTMAX) return NormalizationFunctionImplementation::evaluateSoftmax(values);

    throw std::runtime_error("evaluateNormalizationFunction: unhandled normalizationFunction");
};

Matrix normalizationFunctionDerivative(NormalizationFunction normalizationFunction, const Matrix& values, const Matrix& normalizedValues)
{
    if (normalizationFunction == IDENTITY) return NormalizationFunctionImplementation::identityDerivative(values, normalizedValues);
    if (normalizationFunction == SOFTMAX) return NormalizationFunctionImplementation::softmaxDerivative(values, normalizedValues);

    throw std::runtime_error("normalizationFunctionDerivative: unhandled normalizationFunction");
};

// loss functions

float LossFunctionImplementation::evaluateMeanSquaredError(const Matrix& predictedValues, const Matrix& expectedValues)
{
    auto output = 0.0;

    for (int r = 0;r<predictedValues.rowCount();r++) output += (predictedValues.get(r, 0) - expectedValues.get(r, 0)) * (predictedValues.get(r, 0) - expectedValues.get(r, 0));

    output /= predictedValues.rowCount();

    return output;
};

Matrix LossFunctionImplementation::meanSquaredErrorDerivative(const Matrix& predictedValues, const Matrix& expectedValues)
{
    Matrix output(predictedValues.shape());

    auto coeff = 2.0 / predictedValues.rowCount();

    for (int r = 0;r<predictedValues.rowCount();r++) output.set(r, 0, coeff * (predictedValues.get(r, 0) - expectedValues.get(r, 0)));

    return output;
};

float LossFunctionImplementation::evaluateSumSquaredError(const Matrix& predictedValues, const Matrix& expectedValues)
{
    auto output = 0.0;

    for (int r = 0;r<predictedValues.rowCount();r++) output += (predictedValues.get(r, 0) - expectedValues.get(r, 0)) * (predictedValues.get(r, 0) - expectedValues.get(r, 0));

    return output;
};

Matrix LossFunctionImplementation::sumSquaredErrorDerivative(const Matrix& predictedValues, const Matrix& expectedValues)
{
    Matrix output(predictedValues.shape());

    for (int r = 0;r<predictedValues.rowCount();r++) output.set(r, 0, 2.0 * (predictedValues.get(r, 0) - expectedValues.get(r, 0)));

    return output;
};

float LossFunctionImplementation::evaluateBinaryCrossEntropy(const Matrix& predictedValues, const Matrix& expectedValues)
{
    auto output = 0.0;

    for (int r = 0;r<predictedValues.rowCount();r++) {
        auto predictedValue = std::min(std::max(predictedValues.get(r, 0), 1e-7f), 1.0f - 1e-7f);
        auto expectedValue = expectedValues.get(r, 0);

        output -= expectedValue * log(predictedValue) + (1.0 - expectedValue) * log(1.0 - predictedValue);
    }

    return output;
}

Matrix LossFunctionImplementation::binaryCrossEntropyDerivative(const Matrix& predictedValues, const Matrix& expectedValues)
{
    Matrix output(predictedValues.shape());

    for (int r = 0;r<predictedValues.rowCount();r++) {
        auto predictedValue = std::min(std::max(predictedValues.get(r, 0), 1e-7f), 1.0f - 1e-7f);
        auto expectedValue = expectedValues.get(r, 0);

        output.set(r, 0, (predictedValue - expectedValue) / (predictedValue * (1.0 - predictedValue)));
    }

    return output;
}

float LossFunctionImplementation::evaluateCategoricalCrossEntropy(const Matrix& predictedValues, const Matrix& expectedValues)
{
    auto output = 0.0;

    for (int r = 0;r<predictedValues.rowCount();r++) {
        auto predictedValue = std::min(std::max(predictedValues.get(r, 0), 1e-7f), 1.0f);
        auto expectedValue = expectedValues.get(r, 0);

        output -= expectedValue * log(predictedValue);
    }

    return output;
}

Matrix LossFunctionImplementation::categoricalCrossEntropyDerivative(const Matrix& predictedValues, const Matrix& expectedValues)
{
    Matrix output(predictedValues.shape());

    for (int r = 0;r < predictedValues.rowCount();r++) {
        auto predictedValue = std::min(std::max(predictedValues.get(r, 0), 1e-7f), 1.0f);
        auto expectedValue = expectedValues.get(r, 0);

        output.set(r, 0, -expectedValue / predictedValue);
    }

    return output;
}

float evaluateLossFunction(LossFunction lossFunction, const Matrix& predictedValues, const Matrix& expectedValues)
{
    if (lossFunction == MEAN_SQUARED_ERROR) return LossFunctionImplementation::evaluateMeanSquaredError(predictedValues, expectedValues);
    if (lossFunction == SUM_SQUARED_ERROR) return LossFunctionImplementation::evaluateSumSquaredError(predictedValues, expectedValues);
    if (lossFunction == BINARY_CROSS_ENTROPY) return LossFunctionImplementation::evaluateBinaryCrossEntropy(predictedValues, expectedValues);
    if (lossFunction == CATEGORICAL_CROSS_ENTROPY) return LossFunctionImplementation::evaluateCategoricalCrossEntropy(predictedValues, expectedValues);

    throw std::runtime_error("evaluateLossFunction: unhandled lossFunction");
};

Matrix lossFunctionDerivative(LossFunction lossFunction, const Matrix& predictedValues, const Matrix& expectedValues)
{
    if (lossFunction == MEAN_SQUARED_ERROR) return LossFunctionImplementation::meanSquaredErrorDerivative(predictedValues, expectedValues);
    if (lossFunction == SUM_SQUARED_ERROR) return LossFunctionImplementation::sumSquaredErrorDerivative(predictedValues, expectedValues);
    if (lossFunction == BINARY_CROSS_ENTROPY) return LossFunctionImplementation::binaryCrossEntropyDerivative(predictedValues, expectedValues);
    if (lossFunction == CATEGORICAL_CROSS_ENTROPY) return LossFunctionImplementation::categoricalCrossEntropyDerivative(predictedValues, expectedValues);

    throw std::runtime_error("lossFunctionDerivative: unhandled lossFunction");
};

// hidden layer

HiddenLayerParameters::HiddenLayerParameters(int nodeCount, UnaryActivationFunction unaryActivationFunction)
{
    if (nodeCount < 1) throw std::runtime_error("HiddenLayerParameters constructor: nodeCount is invalid");

    this->nodeCount = nodeCount;
    this->unaryActivationFunction = unaryActivationFunction;

    // random weights and biases are assigned in NeuralNetwork::initializeRandomLayerParameters
};

HiddenLayerParameters::HiddenLayerParameters(UnaryActivationFunction unaryActivationFunction, const Matrix& weights, const Matrix& bias)
{
    if (weights.empty()) throw std::runtime_error("HiddenLayerParameters constructor: weights matrix is empty");
    if (bias.empty()) throw std::runtime_error("HiddenLayerParameters constructor: bias matrix is empty");
    
    if (weights.rowCount() != bias.rowCount()) throw std::runtime_error("HiddenLayerParameters constructor: inconsistent row count between weights matrix and bias matrix");
    if (bias.colCount() != 1) throw std::runtime_error("HiddenLayerParameters constructor: bias matrix is not a column vector");

    this->nodeCount = weights.rowCount();
    this->unaryActivationFunction = unaryActivationFunction;

    this->weights = weights;
    this->bias = bias;
};

// network loss partials

void NetworkLossPartials::add(const NetworkLossPartials& other)
{
    if (this->inputLayerLossPartials.rowCount() != other.inputLayerLossPartials.rowCount()) throw std::runtime_error("NetworkLossPartials add: other has a different number of input nodes");
    if (this->hiddenLayersLossPartials.size() != other.hiddenLayersLossPartials.size()) throw std::runtime_error("NetworkLossPartials add: other has a different number of hidden layers");

    this->inputLayerLossPartials = Matrix::add(this->inputLayerLossPartials, other.inputLayerLossPartials);

    for (int i = 0;i<this->hiddenLayersLossPartials.size();i++) {
        hiddenLayersLossPartials[i].weights = Matrix::add(hiddenLayersLossPartials[i].weights, other.hiddenLayersLossPartials[i].weights);
        hiddenLayersLossPartials[i].bias = Matrix::add(hiddenLayersLossPartials[i].bias, other.hiddenLayersLossPartials[i].bias);
    }
};

void NetworkLossPartials::scalarMultiply(float scalar)
{
    this->inputLayerLossPartials = Matrix::scalarProduct(this->inputLayerLossPartials, scalar);

    for (auto& hiddenLayerLossPartials : this->hiddenLayersLossPartials) {
        hiddenLayerLossPartials.weights = Matrix::scalarProduct(hiddenLayerLossPartials.weights, scalar);
        hiddenLayerLossPartials.bias = Matrix::scalarProduct(hiddenLayerLossPartials.bias, scalar);
    }
};

// neural network

NeuralNetwork::NeuralNetwork(int inputLayerNodeCount, std::vector<HiddenLayerParameters> hiddenLayerParameters, NormalizationFunction outputNormalizationFunction, LossFunction lossFunction)
{
    if (inputLayerNodeCount < 1) throw std::runtime_error("NeuralNetwork constructor: inputLayerNodeCount is invalid");
    if (hiddenLayerParameters.empty()) throw std::runtime_error("NeuralNetwork constructor: hiddenLayerParameters is empty");

    this->inputLayerNodeCount = inputLayerNodeCount;
    this->hiddenLayerStates = std::vector<HiddenLayerState>(hiddenLayerParameters.size());
    this->hiddenLayerParameters = hiddenLayerParameters;
    this->outputNormalizationFunction = outputNormalizationFunction;
    this->lossFunction = lossFunction;
};

std::vector<HiddenLayerState> NeuralNetwork::getHiddenLayerStates()
{
    return this->hiddenLayerStates;
};

std::vector<HiddenLayerParameters> NeuralNetwork::getHiddenLayerParameters()
{
    return this->hiddenLayerParameters;
};

Matrix NeuralNetwork::getNormalizedOutput()
{
    return this->normalizedOutput;
};

LossFunction NeuralNetwork::getLossFunction()
{
    return this->lossFunction;
};

void NeuralNetwork::initializeRandomLayerParameters()
{
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_real_distribution<float> initialWeightDistribution(HiddenLayerParameters::defaultMinInitialWeight, HiddenLayerParameters::defaultMaxInitialWeight);
    std::uniform_real_distribution<float> initialBiasDistribution(HiddenLayerParameters::defaultMinInitialBias, HiddenLayerParameters::defaultMaxInitialBias);

    this->hiddenLayerParameters[0].weights = Matrix(Shape(this->hiddenLayerParameters[0].nodeCount, this->inputLayerNodeCount), rng, initialWeightDistribution);
    this->hiddenLayerParameters[0].bias = Matrix(Shape(this->hiddenLayerParameters[0].nodeCount, 1), rng, initialBiasDistribution);

    for (int i = 1;i<this->hiddenLayerParameters.size();i++) {
        this->hiddenLayerParameters[i].weights = Matrix(Shape(this->hiddenLayerParameters[i].nodeCount, this->hiddenLayerParameters[i - 1].nodeCount), rng, initialWeightDistribution);
        this->hiddenLayerParameters[i].bias = Matrix(Shape(this->hiddenLayerParameters[i].nodeCount, 1), rng, initialBiasDistribution);
    }
};

void NeuralNetwork::initializeRandomLayerParameters(float minInitialWeight, float maxInitialWeight, float minInitialBias, float maxInitialBias)
{
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_real_distribution<float> initialWeightDistribution(minInitialWeight, maxInitialWeight);
    std::uniform_real_distribution<float> initialBiasDistribution(minInitialBias, maxInitialBias);

    this->hiddenLayerParameters[0].weights = Matrix(Shape(this->hiddenLayerParameters[0].nodeCount, this->inputLayerNodeCount), rng, initialWeightDistribution);
    this->hiddenLayerParameters[0].bias = Matrix(Shape(this->hiddenLayerParameters[0].nodeCount, 1), rng, initialBiasDistribution);

    for (int i = 1;i<this->hiddenLayerParameters.size();i++) {
        this->hiddenLayerParameters[i].weights = Matrix(Shape(this->hiddenLayerParameters[i].nodeCount, this->hiddenLayerParameters[i - 1].nodeCount), rng, initialWeightDistribution);
        this->hiddenLayerParameters[i].bias = Matrix(Shape(this->hiddenLayerParameters[i].nodeCount, 1), rng, initialBiasDistribution);
    }
};

void NeuralNetwork::runHiddenLayerFeedForward(int hiddenLayerIndex, const Matrix& input)
{
    auto& hiddenLayerState = this->hiddenLayerStates[hiddenLayerIndex];
    auto& hiddenLayerParameters = this->hiddenLayerParameters[hiddenLayerIndex];

    hiddenLayerState.input = input;
    hiddenLayerState.weighted = Matrix::matrixColumnProduct(hiddenLayerParameters.weights, hiddenLayerState.input);
    hiddenLayerState.biased = Matrix::add(hiddenLayerState.weighted, hiddenLayerParameters.bias);

    hiddenLayerState.activated = evaluateUnaryActivationFunction(hiddenLayerParameters.unaryActivationFunction, hiddenLayerState.biased);
};

Matrix NeuralNetwork::calculateFeedForwardOutput(const Matrix& input)
{
    if (input.rowCount() != this->inputLayerNodeCount) throw std::runtime_error("NeuralNetwork feedForwardOutput: input row count is the wrong size");
    if (input.colCount() != 1) throw std::runtime_error("NeuralNetwork feedForwardOutput: input matrix should be a column vector");

    this->runHiddenLayerFeedForward(0, input);

    for (int i = 1;i<hiddenLayerStates.size();i++) this->runHiddenLayerFeedForward(i, this->hiddenLayerStates[i - 1].activated);

    this->normalizedOutput = evaluateNormalizationFunction(this->outputNormalizationFunction, this->hiddenLayerStates.back().activated);

    return this->normalizedOutput;
};

float NeuralNetwork::calculateLoss(const Matrix& input, const Matrix& expectedOutput)
{
    if (expectedOutput.rowCount() != this->hiddenLayerParameters.back().nodeCount) throw std::runtime_error("NeuralNetwork calculateLoss: incorrect number of expected outputs");

    auto predictedValues = this->calculateFeedForwardOutput(input);

    return evaluateLossFunction(this->lossFunction, predictedValues, expectedOutput);
};

void NeuralNetwork::calculateHiddenLayerLossPartials(int hiddenLayerIndex, const Matrix& dLossWrtActivated)
{
    auto& hiddenLayerState = this->hiddenLayerStates[hiddenLayerIndex];
    auto& hiddenLayerParameters = this->hiddenLayerParameters[hiddenLayerIndex];

    hiddenLayerState.dLossWrtActivated = dLossWrtActivated;

    auto dActivatedWrtBiased = unaryActivationFunctionDerivative(hiddenLayerParameters.unaryActivationFunction, hiddenLayerState.biased, hiddenLayerState.activated);

    hiddenLayerState.dLossWrtBiased = Matrix::hadamardProduct(hiddenLayerState.dLossWrtActivated, dActivatedWrtBiased);

    // this has different semantic meaning than dLossWrtBiased, technically could be consolidated
    hiddenLayerState.dLossWrtBias = hiddenLayerState.dLossWrtBiased;

    hiddenLayerState.dLossWrtWeights = Matrix(hiddenLayerParameters.weights.shape());

    for (int i = 0;i<hiddenLayerParameters.weights.rowCount();i++) {
        for (int j = 0;j<hiddenLayerParameters.weights.colCount();j++) {
            hiddenLayerState.dLossWrtWeights.set(i, j, hiddenLayerState.dLossWrtBiased.get(i, 0) * hiddenLayerState.input.get(j, 0));
        }
    }

    hiddenLayerState.dLossWrtInput = Matrix::matrixProduct(Matrix::transpose(hiddenLayerParameters.weights), hiddenLayerState.dLossWrtBiased);
};

NetworkLossPartials NeuralNetwork::calculateNetworkLossPartials(const Matrix& expectedOutput)
{
    if (expectedOutput.rowCount() != this->hiddenLayerParameters.back().nodeCount) throw std::runtime_error("NeuralNetwork calculateBackPropagationAdjustments: incorrect number of expected outputs");

    auto dLossWrtNormalizedOutput = lossFunctionDerivative(this->lossFunction, this->normalizedOutput, expectedOutput);

    auto dNormalizedOutputWrtActivated = normalizationFunctionDerivative(this->outputNormalizationFunction, this->hiddenLayerStates.back().activated, this->normalizedOutput);

    auto dLossWrtActivated = Matrix::matrixColumnProduct(dNormalizedOutputWrtActivated, dLossWrtNormalizedOutput);

    this->calculateHiddenLayerLossPartials(this->hiddenLayerStates.size() - 1, dLossWrtActivated);

    for (int i = this->hiddenLayerStates.size() - 2;i>=0;i--) this->calculateHiddenLayerLossPartials(i, this->hiddenLayerStates[i + 1].dLossWrtInput);

    std::vector<HiddenLayerLossPartials> hiddenLayerLossPartials(this->hiddenLayerStates.size());

    for (int i = 0;i<this->hiddenLayerStates.size();i++) hiddenLayerLossPartials[i] = HiddenLayerLossPartials(
        this->hiddenLayerStates[i].dLossWrtWeights,
        this->hiddenLayerStates[i].dLossWrtBias
    );

    Matrix inputLayerLossPartials = this->hiddenLayerStates[0].dLossWrtInput;

    return NetworkLossPartials(inputLayerLossPartials, hiddenLayerLossPartials);
};

NetworkLossPartials NeuralNetwork::train(DataPoint trainingDataPoint, float learningRate)
{
    this->calculateFeedForwardOutput(trainingDataPoint.input);

    auto networkLossPartials = this->calculateNetworkLossPartials(trainingDataPoint.expectedOutput);

    auto parameterAdjustments = networkLossPartials;
    parameterAdjustments.scalarMultiply(-learningRate);

    for (int i = 0;i<this->hiddenLayerParameters.size();i++) {
        this->hiddenLayerParameters[i].weights = Matrix::add(this->hiddenLayerParameters[i].weights, parameterAdjustments.hiddenLayersLossPartials[i].weights);
        this->hiddenLayerParameters[i].bias = Matrix::add(this->hiddenLayerParameters[i].bias, parameterAdjustments.hiddenLayersLossPartials[i].bias);
    }

    return networkLossPartials;
};

void NeuralNetwork::batchTrain(std::vector<DataPoint> trainingDataBatch, float learningRate)
{
    if (trainingDataBatch.empty()) throw std::runtime_error("NeuralNetwork batchTrain: trainingDataBatch is empty");

    this->calculateFeedForwardOutput(trainingDataBatch[0].input);
    NetworkLossPartials averageLossPartials = this->calculateNetworkLossPartials(trainingDataBatch[0].expectedOutput);

    for (int i = 1;i<trainingDataBatch.size();i++) {
        this->calculateFeedForwardOutput(trainingDataBatch[i].input);

        averageLossPartials.add(this->calculateNetworkLossPartials(trainingDataBatch[i].expectedOutput));
    }

    averageLossPartials.scalarMultiply(-learningRate / trainingDataBatch.size());

    for (int i = 0;i<this->hiddenLayerParameters.size();i++) {
        this->hiddenLayerParameters[i].weights = Matrix::add(this->hiddenLayerParameters[i].weights, averageLossPartials.hiddenLayersLossPartials[i].weights);
        this->hiddenLayerParameters[i].bias = Matrix::add(this->hiddenLayerParameters[i].bias, averageLossPartials.hiddenLayersLossPartials[i].bias);
    }
};

std::string NeuralNetwork::toString()
{
    std::string output;

    output += "{\n";

    output += "\tInput Layer (" + std::to_string(this->inputLayerNodeCount) + " nodes)\n\n";
    
    for (size_t i = 0;i<hiddenLayerStates.size();i++) {
        auto state = hiddenLayerStates[i];
        auto parameters = hiddenLayerParameters[i];

        output += "\tHidden Layer (" + std::to_string(parameters.nodeCount) + " nodes)\n";

        output += "\t\tWeights: " + parameters.weights.toString() + "\n";
        output += "\t\tBias: " + parameters.bias.toString() + "\n";

        output += "\n";

        output += "\t\tInput: " + state.input.toString() + "\n";
        output += "\t\tWeighted: " + state.weighted.toString() + "\n";
        output += "\t\tBiased: " + state.biased.toString() + "\n";
        output += "\t\tActivated: " + state.activated.toString() + "\n";

        output += "\n";

        output += "\t\tdLossWrtActivated: " + state.dLossWrtActivated.toString() + "\n";
        output += "\t\tdLossWrtBias: " + state.dLossWrtBias.toString() + "\n";
        output += "\t\tdLossWrtWeights: " + state.dLossWrtWeights.toString() + "\n";
        output += "\t\tdLossWrtInput: " + state.dLossWrtInput.toString() + "\n";

        output += "\n";
    }

    output += "\tOutput Layer (" + std::to_string(this->normalizedOutput.rowCount()) + " nodes)\n";

    output += "\t\tActivated: " + this->normalizedOutput.toString() + "\n";

    output += "}";

    return output;
};