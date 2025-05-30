#ifndef NEURAL_NETWORK_HPP
#define NEURAL_NETWORK_HPP

#include "matrix.hpp"

enum UnaryActivationFunction
{
    LINEAR,
    RELU,
    SIGMOID,
    TANH
};

class UnaryActivationFunctionImplementation
{
    public:
        static Matrix evaluateLinear(const Matrix& values);
        static Matrix linearDerivative(const Matrix& values, const Matrix& activatedValues);

        static Matrix evaluateRelu(const Matrix& values);
        static Matrix reluDerivative(const Matrix& values, const Matrix& activatedValues);

        static Matrix evaluateSigmoid(const Matrix& values);
        static Matrix sigmoidDerivative(const Matrix& values, const Matrix& activatedValues);

        static Matrix evaluateTanh(const Matrix& values);
        static Matrix tanhDerivative(const Matrix& values, const Matrix& activatedValues);
};

Matrix evaluateUnaryActivationFunction(UnaryActivationFunction unaryActivationFunction, const Matrix& values);
Matrix unaryActivationFunctionDerivative(UnaryActivationFunction unaryActivationFunction, const Matrix& values, const Matrix& activatedValues);

enum NormalizationFunction
{
    IDENTITY,
    SOFTMAX
};

class NormalizationFunctionImplementation
{
    public:
        static Matrix evaluateIdentity(const Matrix& values);
        static Matrix identityDerivative(const Matrix& values, const Matrix& normalizedValues);

        static Matrix evaluateSoftmax(const Matrix& values);
        static Matrix softmaxDerivative(const Matrix& values, const Matrix& normalizedValues);
};

Matrix evaluateNormalizationFunction(NormalizationFunction lossFunction, const Matrix& values);
Matrix normalizationFunctionDerivative(NormalizationFunction lossFunction, const Matrix& values, const Matrix& normalizedValues);

enum LossFunction
{
    MEAN_SQUARED_ERROR,
    SUM_SQUARED_ERROR,
    BINARY_CROSS_ENTROPY,
    CATEGORICAL_CROSS_ENTROPY
};

class LossFunctionImplementation
{
    public:
        static float evaluateMeanSquaredError(const Matrix& predictedValues, const Matrix& expectedValues);
        static Matrix meanSquaredErrorDerivative(const Matrix& predictedValues, const Matrix& expectedValues);

        static float evaluateSumSquaredError(const Matrix& predictedValues, const Matrix& expectedValues);
        static Matrix sumSquaredErrorDerivative(const Matrix& predictedValues, const Matrix& expectedValues);

        static float evaluateBinaryCrossEntropy(const Matrix& predictedValues, const Matrix& expectedValues);
        static Matrix binaryCrossEntropyDerivative(const Matrix& predictedValues, const Matrix& expectedValues);

        static float evaluateCategoricalCrossEntropy(const Matrix& predictedValues, const Matrix& expectedValues);
        static Matrix categoricalCrossEntropyDerivative(const Matrix& predictedValues, const Matrix& expectedValues);
};

float evaluateLossFunction(LossFunction lossFunction, const Matrix& predictedValues, const Matrix& expectedValues);
Matrix lossFunctionDerivative(LossFunction lossFunction, const Matrix& predictedValues, const Matrix& expectedValues);

#endif