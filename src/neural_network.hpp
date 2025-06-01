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

class HiddenLayerState
{
    public:
        // feed forward stages
        Matrix input;
        Matrix weighted;
        Matrix biased;
        Matrix activated;

        // back propagation stages
        Matrix dLossWrtActivated;
        Matrix dLossWrtBiased;
        Matrix dLossWrtBias;
        Matrix dLossWrtWeights;
        Matrix dLossWrtInput;
};

class HiddenLayerParameters
{
    public:
        int nodeCount;
        UnaryActivationFunction unaryActivationFunction;

        Matrix weights;
        Matrix bias;

        HiddenLayerParameters(int nodeCount, UnaryActivationFunction unaryActivationFunction);
        HiddenLayerParameters(UnaryActivationFunction unaryActivationFunction, const Matrix& weights, const Matrix& bias);

        static constexpr float minInitialWeight = -5.0;
        static constexpr float maxInitialWeight = 5.0;

        static constexpr float minInitialBias = -5.0;
        static constexpr float maxInitialBias = 5.0;
};

class HiddenLayerLossPartials
{
    public:
        Matrix weights;
        Matrix bias;

        HiddenLayerLossPartials() = default;
        HiddenLayerLossPartials(const Matrix& weights, const Matrix& bias): weights(weights), bias(bias) {};
};

class NetworkLossPartials
{
    public:
        Matrix inputLayerLossPartials;
        std::vector<HiddenLayerLossPartials> hiddenLayersLossPartials;

        NetworkLossPartials() = default;
        NetworkLossPartials(const Matrix& inputLayerLossPartials, const std::vector<HiddenLayerLossPartials>& hiddenLayersLossPartials): inputLayerLossPartials(inputLayerLossPartials), hiddenLayersLossPartials(hiddenLayersLossPartials) {};

        void add(const NetworkLossPartials& other);

        void scalarMultiply(float scalar);
};

class DataPoint
{
    public:
        Matrix input;
        Matrix expectedOutput;

        DataPoint(const Matrix& input, const Matrix& expectedOutput): input(input), expectedOutput(expectedOutput) {};
};

class NeuralNetwork
{
    private:
        int inputLayerNodeCount;
        std::vector<HiddenLayerState> hiddenLayerStates;
        std::vector<HiddenLayerParameters> hiddenLayerParameters;
        NormalizationFunction outputNormalizationFunction;
        LossFunction outputLossFunction;
        Matrix normalizedOutput;

        void runHiddenLayerFeedForward(int hiddenLayerIndex, const Matrix& input);

        // should only be called after feed forward has run
        void calculateHiddenLayerLossPartials(int hiddenLayerIndex, const Matrix& dLossWrtActivated);
        NetworkLossPartials calculateNetworkLossPartials(const Matrix& expectedOutput);
    
    public:
        NeuralNetwork(int inputLayerNodeCount, std::vector<HiddenLayerParameters> hiddenLayerParameters, NormalizationFunction outputNormalizationFunction, LossFunction outputLossFunction);

        std::vector<HiddenLayerState> getHiddenLayerStates();
        std::vector<HiddenLayerParameters> getHiddenLayerParameters();
        Matrix getNormalizedOutput();

        void initializeRandomLayerParameters();

        Matrix calculateFeedForwardOutput(const Matrix& input);

        float calculateLoss(const Matrix& input, const Matrix& expectedOutput);

        NetworkLossPartials train(DataPoint trainingDataPoint, float learningRate);

        void batchTrain(std::vector<DataPoint> trainingDataBatch, float learningRate);

        std::string toString();
};

#endif