#include "neural_network.hpp"

#include <xtensor/core/xmath.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/core/xnoalias.hpp>

// unary activation functions

Matrix UnaryActivationFunctionImplementation::evaluateLinear(const Matrix& values)
{
    return values;
};

Matrix UnaryActivationFunctionImplementation::evaluateLinearDerivative(const Matrix& values, const Matrix&)
{
    return xt::full_like(values, 1.0);
};

Matrix UnaryActivationFunctionImplementation::evaluateRelu(const Matrix& values)
{
    return xt::maximum(values, 0.0);
};

Matrix UnaryActivationFunctionImplementation::evaluateReluDerivative(const Matrix& values, const Matrix&)
{
    return xt::where(values > 0.0, 1.0, 0.0);
};

Matrix UnaryActivationFunctionImplementation::evaluateSigmoid(const Matrix& values)
{
    return 1.0 / (1.0 + xt::exp(-values));
};

Matrix UnaryActivationFunctionImplementation::evaluateSigmoidDerivative(const Matrix&, const Matrix& activatedValues)
{
    return activatedValues * (1.0f - activatedValues);
};

Matrix UnaryActivationFunctionImplementation::evaluateTanh(const Matrix& values)
{
    return tanh(values);
};

Matrix UnaryActivationFunctionImplementation::evaluateTanhDerivative(const Matrix&, const Matrix& activatedValues)
{
    return 1.0 - activatedValues * activatedValues;
};

Matrix UnaryActivationFunctionImplementation::evaluateAtan(const Matrix& values)
{
    return atan(values);
};

Matrix UnaryActivationFunctionImplementation::evaluateAtanDerivative(const Matrix&, const Matrix& activatedValues)
{
    return 1.0 / (1.0 + activatedValues * activatedValues);
};

Matrix evaluateUnaryActivation(UnaryActivationFunction unaryActivationFunction, const Matrix& values)
{
    switch (unaryActivationFunction) {
        case LINEAR:
            return UnaryActivationFunctionImplementation::evaluateLinear(values);
        case RELU:
            return UnaryActivationFunctionImplementation::evaluateRelu(values);
        case SIGMOID:
            return UnaryActivationFunctionImplementation::evaluateSigmoid(values);
        case TANH:
            return UnaryActivationFunctionImplementation::evaluateTanh(values);
        case ATAN:
            return UnaryActivationFunctionImplementation::evaluateAtan(values);
        default:
            throw std::runtime_error("evaluateUnaryActivation: unhandled unaryActivationFunction");
    }
};

Matrix evaluateUnaryActivationDerivative(UnaryActivationFunction unaryActivationFunction, const Matrix& values, const Matrix& activatedValues)
{
    switch (unaryActivationFunction) {
        case LINEAR:
            return UnaryActivationFunctionImplementation::evaluateLinearDerivative(values, activatedValues);
        case RELU:
            return UnaryActivationFunctionImplementation::evaluateReluDerivative(values, activatedValues);
        case SIGMOID:
            return UnaryActivationFunctionImplementation::evaluateSigmoidDerivative(values, activatedValues);
        case TANH:
            return UnaryActivationFunctionImplementation::evaluateTanhDerivative(values, activatedValues);
        case ATAN:
            return UnaryActivationFunctionImplementation::evaluateAtanDerivative(values, activatedValues);
        default:
            throw std::runtime_error("evaluateUnaryActivationDerivative: unhandled unaryActivationFunction");
    }
};

// normalization functions

Matrix NormalizationFunctionImplementation::evaluateIdentity(const Matrix& values)
{
    return values;
};

Matrix NormalizationFunctionImplementation::evaluateIdentityDerivative(const Matrix& values, const Matrix&)
{
    return xt::eye<float>(values.shape()[0]);
};

Matrix NormalizationFunctionImplementation::evaluateSoftmax(const Matrix& values)
{
    auto output = xt::exp(values - xt::amax(values)());

    return output / xt::sum(output)();
};

Matrix NormalizationFunctionImplementation::evaluateSoftmaxDerivative(const Matrix&, const Matrix& normalizedValues)
{
    Matrix output = -normalizedValues * xt::transpose(normalizedValues);

    for (int i = 0;i<normalizedValues.shape()[0];i++) output(i, i) += normalizedValues(i, i);

    return output;
};

Matrix evaluateNormalization(NormalizationFunction normalizationFunction, const Matrix& values)
{
    switch (normalizationFunction) {
        case IDENTITY:
            return NormalizationFunctionImplementation::evaluateIdentity(values);
        case SOFTMAX:
            return NormalizationFunctionImplementation::evaluateSoftmax(values);
        default:
            throw std::runtime_error("evaluateNormalization: unhandled normalizationFunction");
    }
};

Matrix evaluateNormalizationDerivative(NormalizationFunction normalizationFunction, const Matrix& values, const Matrix& normalizedValues)
{
    switch (normalizationFunction) {
        case IDENTITY:
            return NormalizationFunctionImplementation::evaluateIdentityDerivative(values, normalizedValues);
        case SOFTMAX:
            return NormalizationFunctionImplementation::evaluateSoftmaxDerivative(values, normalizedValues);
        default:
            throw std::runtime_error("evaluateNormalizationDerivative: unhandled normalizationFunction");
    }
};

// loss functions

float LossFunctionImplementation::evaluateMeanSquaredError(const Matrix& predictedValues, const Matrix& expectedValues)
{
    return xt::mean((predictedValues - expectedValues) * (predictedValues - expectedValues))();
};

Matrix LossFunctionImplementation::evaluateMeanSquaredErrorDerivative(const Matrix& predictedValues, const Matrix& expectedValues)
{
    return 2.0 * (predictedValues - expectedValues) / predictedValues.size();
};

float LossFunctionImplementation::evaluateSumSquaredError(const Matrix& predictedValues, const Matrix& expectedValues)
{
    return xt::sum((predictedValues - expectedValues) * (predictedValues - expectedValues))();
};

Matrix LossFunctionImplementation::evaluateSumSquaredErrorDerivative(const Matrix& predictedValues, const Matrix& expectedValues)
{
    return 2.0 * (predictedValues - expectedValues);
};

float LossFunctionImplementation::evaluateBinaryCrossEntropy(const Matrix& predictedValues, const Matrix& expectedValues)
{
    auto clampedPredictedValues = xt::clip(predictedValues, 1e-7, 1.0 - 1e-7);

    return -xt::sum(expectedValues * log(clampedPredictedValues) + (1.0 - expectedValues) * log(1.0 - clampedPredictedValues))();
}

Matrix LossFunctionImplementation::evaluateBinaryCrossEntropyDerivative(const Matrix& predictedValues, const Matrix& expectedValues)
{
    auto clampedPredictedValues = xt::clip(predictedValues, 1e-7, 1.0 - 1e-7);

    return (clampedPredictedValues - expectedValues) / (clampedPredictedValues * (1.0 - clampedPredictedValues));
}

float LossFunctionImplementation::evaluateCategoricalCrossEntropy(const Matrix& predictedValues, const Matrix& expectedValues)
{
    auto clampedPredictedValues = xt::clip(predictedValues, 1e-7, 1.0 - 1e-7);

    return -xt::sum(expectedValues * log(clampedPredictedValues))();
}

Matrix LossFunctionImplementation::evaluateCategoricalCrossEntropyDerivative(const Matrix& predictedValues, const Matrix& expectedValues)
{
    auto clampedPredictedValues = xt::clip(predictedValues, 1e-7, 1.0 - 1e-7);

    return -expectedValues / clampedPredictedValues;
}

float evaluateLoss(LossFunction lossFunction, const Matrix& predictedValues, const Matrix& expectedValues)
{
    switch (lossFunction) {
        case MEAN_SQUARED_ERROR:
            return LossFunctionImplementation::evaluateMeanSquaredError(predictedValues, expectedValues);
        case SUM_SQUARED_ERROR:
            return LossFunctionImplementation::evaluateSumSquaredError(predictedValues, expectedValues);
        case BINARY_CROSS_ENTROPY:
            return LossFunctionImplementation::evaluateBinaryCrossEntropy(predictedValues, expectedValues);
        case CATEGORICAL_CROSS_ENTROPY:
            return LossFunctionImplementation::evaluateCategoricalCrossEntropy(predictedValues, expectedValues);
        default:
            throw std::runtime_error("evaluateLoss: unhandled lossFunction");
    }
};

Matrix evaluateLossDerivative(LossFunction lossFunction, const Matrix& predictedValues, const Matrix& expectedValues)
{
    switch (lossFunction) {
        case MEAN_SQUARED_ERROR:
            return LossFunctionImplementation::evaluateMeanSquaredErrorDerivative(predictedValues, expectedValues);
        case SUM_SQUARED_ERROR:
            return LossFunctionImplementation::evaluateSumSquaredErrorDerivative(predictedValues, expectedValues);
        case BINARY_CROSS_ENTROPY:
            return LossFunctionImplementation::evaluateBinaryCrossEntropyDerivative(predictedValues, expectedValues);
        case CATEGORICAL_CROSS_ENTROPY:
            return LossFunctionImplementation::evaluateCategoricalCrossEntropyDerivative(predictedValues, expectedValues);
        default:
            throw std::runtime_error("evaluateLossDerivative: unhandled lossFunction");
    }
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
    if (weights.size() == 0) throw std::runtime_error("HiddenLayerParameters constructor: weights matrix is empty");
    if (bias.size() == 0) throw std::runtime_error("HiddenLayerParameters constructor: bias matrix is empty");
    
    if (weights.shape()[0] != bias.shape()[0]) throw std::runtime_error("HiddenLayerParameters constructor: inconsistent row count between weights matrix and bias matrix");
    if (bias.shape()[1] != 1) throw std::runtime_error("HiddenLayerParameters constructor: bias matrix is not a column vector");

    this->nodeCount = weights.shape()[0];
    this->unaryActivationFunction = unaryActivationFunction;

    this->weights = weights;
    this->bias = bias;
};

// network loss partials

void NetworkLossPartials::add(const NetworkLossPartials& other)
{
    if (this->inputLayerLossPartials.shape()[0] != other.inputLayerLossPartials.shape()[0]) throw std::runtime_error("NetworkLossPartials add: other has a different number of input nodes");
    if (this->hiddenLayersLossPartials.size() != other.hiddenLayersLossPartials.size()) throw std::runtime_error("NetworkLossPartials add: other has a different number of hidden layers");

    this->loss += other.loss;

    xt::noalias(this->inputLayerLossPartials) += other.inputLayerLossPartials;

    for (int i = 0;i<this->hiddenLayersLossPartials.size();i++) {
        xt::noalias(hiddenLayersLossPartials[i].weights) += other.hiddenLayersLossPartials[i].weights;
        xt::noalias(hiddenLayersLossPartials[i].bias) += other.hiddenLayersLossPartials[i].bias;
    }
};

void NetworkLossPartials::scalarMultiply(float scalar)
{
    this->loss *= scalar;

    xt::noalias(this->inputLayerLossPartials) *= scalar;

    for (auto& hiddenLayerLossPartials : this->hiddenLayersLossPartials) {
        xt::noalias(hiddenLayerLossPartials.weights) *= scalar;
        xt::noalias(hiddenLayerLossPartials.bias) *= scalar;
    }
};

// neural network

NeuralNetwork::NeuralNetwork(int inputLayerNodeCount, std::vector<HiddenLayerParameters> hiddenLayerParameters, NormalizationFunction outputNormalizationFunction, LossFunction lossFunction)
{
    if (inputLayerNodeCount < 1) throw std::runtime_error("NeuralNetwork constructor: inputLayerNodeCount is invalid");
    if (hiddenLayerParameters.empty()) throw std::runtime_error("NeuralNetwork constructor: hiddenLayerParameters is empty");

    this->inputLayerNodeCount = inputLayerNodeCount;
    this->hiddenLayerStates.resize(hiddenLayerParameters.size());
    this->hiddenLayerParameters = hiddenLayerParameters;
    this->outputNormalizationFunction = outputNormalizationFunction;
    this->lossFunction = lossFunction;
};

int NeuralNetwork::getInputLayerNodeCount()
{
    return this->inputLayerNodeCount;
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

void NeuralNetwork::initializeRandomHiddenLayerParameters()
{
    this->initializeRandomHiddenLayerParameters(HiddenLayerParameters::defaultMinInitialWeight, HiddenLayerParameters::defaultMaxInitialWeight, HiddenLayerParameters::defaultMinInitialBias, HiddenLayerParameters::defaultMaxInitialBias);
};

void NeuralNetwork::initializeRandomHiddenLayerParameters(float minInitialWeight, float maxInitialWeight, float minInitialBias, float maxInitialBias)
{
    this->hiddenLayerParameters[0].weights = randomMatrix(this->hiddenLayerParameters[0].nodeCount, this->inputLayerNodeCount, minInitialWeight, maxInitialWeight);
    this->hiddenLayerParameters[0].bias = randomMatrix(this->hiddenLayerParameters[0].nodeCount, 1, minInitialBias, maxInitialBias);

    for (int i = 1;i<this->hiddenLayerParameters.size();i++) {
        this->hiddenLayerParameters[i].weights = randomMatrix(this->hiddenLayerParameters[i].nodeCount, this->hiddenLayerParameters[i - 1].nodeCount, minInitialWeight, maxInitialWeight);
        this->hiddenLayerParameters[i].bias = randomMatrix(this->hiddenLayerParameters[i].nodeCount, 1, minInitialBias, maxInitialBias);
    }
};

void NeuralNetwork::runHiddenLayerFeedForward(int hiddenLayerIndex, const Matrix& input)
{
    auto& hiddenLayerState = this->hiddenLayerStates[hiddenLayerIndex];
    auto& hiddenLayerParameters = this->hiddenLayerParameters[hiddenLayerIndex];

    hiddenLayerState.input = input;
    hiddenLayerState.weighted = xt::linalg::dot(hiddenLayerParameters.weights, hiddenLayerState.input);
    hiddenLayerState.biased = hiddenLayerState.weighted + hiddenLayerParameters.bias;

    hiddenLayerState.activated = evaluateUnaryActivation(hiddenLayerParameters.unaryActivationFunction, hiddenLayerState.biased);
};

Matrix NeuralNetwork::calculateFeedForwardOutput(const Matrix& input)
{
    if (input.shape()[0] != this->inputLayerNodeCount) throw std::runtime_error("NeuralNetwork feedForwardOutput: input row count is the wrong size");
    if (input.shape()[1] != 1) throw std::runtime_error("NeuralNetwork feedForwardOutput: input matrix should be a column vector");

    this->runHiddenLayerFeedForward(0, input);

    for (int i = 1;i<hiddenLayerStates.size();i++) this->runHiddenLayerFeedForward(i, this->hiddenLayerStates[i - 1].activated);

    this->normalizedOutput = evaluateNormalization(this->outputNormalizationFunction, this->hiddenLayerStates.back().activated);

    return this->normalizedOutput;
};

float NeuralNetwork::calculateLoss(const Matrix& expectedOutput)
{
    auto predictedValues = this->getNormalizedOutput();

    return evaluateLoss(this->lossFunction, predictedValues, expectedOutput);
};

float NeuralNetwork::calculateLoss(const Matrix& input, const Matrix& expectedOutput)
{
    if (expectedOutput.shape()[0] != this->hiddenLayerParameters.back().nodeCount) throw std::runtime_error("NeuralNetwork calculateLoss: incorrect number of expected outputs");

    auto predictedValues = this->calculateFeedForwardOutput(input);

    return evaluateLoss(this->lossFunction, predictedValues, expectedOutput);
};

void NeuralNetwork::calculateHiddenLayerLossPartials(int hiddenLayerIndex, const Matrix& dLossWrtActivated)
{
    auto& hiddenLayerState = this->hiddenLayerStates[hiddenLayerIndex];
    auto& hiddenLayerParameters = this->hiddenLayerParameters[hiddenLayerIndex];

    xt::noalias(hiddenLayerState.dLossWrtActivated) = dLossWrtActivated;

    xt::noalias(hiddenLayerState.dLossWrtBiased) = hiddenLayerState.dLossWrtActivated * evaluateUnaryActivationDerivative(hiddenLayerParameters.unaryActivationFunction, hiddenLayerState.biased, hiddenLayerState.activated);

    xt::noalias(hiddenLayerState.dLossWrtWeights) = hiddenLayerState.dLossWrtBiased * xt::transpose(hiddenLayerState.input);

    xt::noalias(hiddenLayerState.dLossWrtInput) = xt::linalg::dot(xt::transpose(hiddenLayerParameters.weights), hiddenLayerState.dLossWrtBiased);
};

NetworkLossPartials NeuralNetwork::calculateLossPartials(const Matrix& expectedOutput)
{
    if (expectedOutput.shape()[0] != this->hiddenLayerParameters.back().nodeCount) throw std::runtime_error("NeuralNetwork calculateBackPropagationAdjustments: incorrect number of expected outputs");

    auto dLossWrtNormalizedOutput = evaluateLossDerivative(this->lossFunction, this->normalizedOutput, expectedOutput);

    auto dNormalizedOutputWrtActivated = evaluateNormalizationDerivative(this->outputNormalizationFunction, this->hiddenLayerStates.back().activated, this->normalizedOutput);

    auto dLossWrtActivated = xt::linalg::dot(dNormalizedOutputWrtActivated, dLossWrtNormalizedOutput);

    this->calculateHiddenLayerLossPartials(this->hiddenLayerStates.size() - 1, dLossWrtActivated);

    for (int i = this->hiddenLayerStates.size() - 2;i>=0;i--) this->calculateHiddenLayerLossPartials(i, this->hiddenLayerStates[i + 1].dLossWrtInput);

    std::vector<HiddenLayerLossPartials> hiddenLayerLossPartials(this->hiddenLayerStates.size());

    for (int i = 0;i<this->hiddenLayerStates.size();i++) hiddenLayerLossPartials[i] = HiddenLayerLossPartials(
        this->hiddenLayerStates[i].dLossWrtWeights,
        this->hiddenLayerStates[i].dLossWrtBiased
    );

    Matrix inputLayerLossPartials = this->hiddenLayerStates[0].dLossWrtInput;

    return NetworkLossPartials(this->calculateLoss(expectedOutput), inputLayerLossPartials, hiddenLayerLossPartials);
};

NetworkLossPartials NeuralNetwork::calculateLossPartials(DataPoint dataPoint)
{
    this->calculateFeedForwardOutput(dataPoint.input);

    return this->calculateLossPartials(dataPoint.expectedOutput);
};

NetworkLossPartials NeuralNetwork::calculateBatchLossPartials(std::vector<DataPoint> dataBatch)
{
    if (dataBatch.empty()) throw std::runtime_error("NeuralNetwork calculateBatchLossPartials: dataBatch is empty");

    auto averageLossPartials = this->calculateLossPartials(dataBatch[0]);

    for (int i = 1;i<dataBatch.size();i++) averageLossPartials.add(this->calculateLossPartials(dataBatch[i]));
    
    averageLossPartials.scalarMultiply(1.0 / dataBatch.size());
    
    return averageLossPartials;
};

void NeuralNetwork::applyLossPartials(NetworkLossPartials lossPartials)
{
    if (this->hiddenLayerParameters.size() != lossPartials.hiddenLayersLossPartials.size()) throw std::runtime_error("NeuralNetwork applyLossPartials: lossPartials has different number of hidden layers");

    for (int i = 0;i<this->hiddenLayerParameters.size();i++) {
        xt::noalias(this->hiddenLayerParameters[i].weights) += lossPartials.hiddenLayersLossPartials[i].weights;
        xt::noalias(this->hiddenLayerParameters[i].bias) += lossPartials.hiddenLayersLossPartials[i].bias;
    }
};

void NeuralNetwork::train(DataPoint trainingDataPoint, float learningRate)
{
    auto parameterAdjustments = this->calculateLossPartials(trainingDataPoint);
    parameterAdjustments.scalarMultiply(-learningRate);

    this->applyLossPartials(parameterAdjustments);
};

void NeuralNetwork::batchTrain(std::vector<DataPoint> trainingDataBatch, float learningRate)
{
    auto parameterAdjustments = this->calculateBatchLossPartials(trainingDataBatch);
    parameterAdjustments.scalarMultiply(-learningRate);
    
    this->applyLossPartials(parameterAdjustments);
};

std::string NeuralNetwork::toString() const
{
    std::string output;

    output += "{\n";

    output += "\tInput Layer (" + std::to_string(this->inputLayerNodeCount) + " nodes)\n\n";
    
    for (int i = 0;i<this->hiddenLayerStates.size();i++) {
        auto state = this->hiddenLayerStates[i];
        auto parameters = this->hiddenLayerParameters[i];

        output += "\tHidden Layer (" + std::to_string(parameters.nodeCount) + " nodes)\n";

        output += "\t\tWeights: " + matrixToStr(parameters.weights) + "\n";
        output += "\t\tBias: " + matrixToStr(parameters.bias) + "\n";

        output += "\n";

        output += "\t\tInput: " + matrixToStr(state.input) + "\n";
        output += "\t\tWeighted: " + matrixToStr(state.weighted) + "\n";
        output += "\t\tBiased: " + matrixToStr(state.biased) + "\n";
        output += "\t\tActivated: " + matrixToStr(state.activated) + "\n";

        output += "\n";

        output += "\t\tdLossWrtActivated: " + matrixToStr(state.dLossWrtActivated) + "\n";
        output += "\t\tdLossWrtBiased: " + matrixToStr(state.dLossWrtBiased) + "\n";
        output += "\t\tdLossWrtWeights: " + matrixToStr(state.dLossWrtWeights) + "\n";
        output += "\t\tdLossWrtInput: " + matrixToStr(state.dLossWrtInput) + "\n";

        output += "\n";
    }

    output += "\tOutput Layer (" + std::to_string(this->normalizedOutput.shape()[0]) + " nodes)\n";

    output += "\t\tActivated: " + matrixToStr(this->normalizedOutput) + "\n";

    output += "}";

    return output;
};