#ifndef NEURAL_NETWORK_HPP
#define NEURAL_NETWORK_HPP

// #include <cereal/archives/json.hpp>
// #include <cereal/types/vector.hpp>

#include "matrix.hpp"

enum UnaryActivationFunction
{
    LINEAR,
    RELU,
    SIGMOID,
    TANH,
    ATAN
};

class UnaryActivationFunctionImplementation
{
    public:
        static Matrix evaluateLinear(const Matrix& values);
        static Matrix evaluateLinearDerivative(const Matrix& values, const Matrix& activatedValues);

        static Matrix evaluateRelu(const Matrix& values);
        static Matrix evaluateReluDerivative(const Matrix& values, const Matrix& activatedValues);

        static Matrix evaluateSigmoid(const Matrix& values);
        static Matrix evaluateSigmoidDerivative(const Matrix& values, const Matrix& activatedValues);

        static Matrix evaluateTanh(const Matrix& values);
        static Matrix evaluateTanhDerivative(const Matrix& values, const Matrix& activatedValues);

        static Matrix evaluateAtan(const Matrix& values);
        static Matrix evaluateAtanDerivative(const Matrix& values, const Matrix& activatedValues);
};

Matrix evaluateUnaryActivation(UnaryActivationFunction unaryActivationFunction, const Matrix& values);
Matrix evaluateUnaryActivationDerivative(UnaryActivationFunction unaryActivationFunction, const Matrix& values, const Matrix& activatedValues);

enum NormalizationFunction
{
    IDENTITY,
    SOFTMAX
};

class NormalizationFunctionImplementation
{
    public:
        static Matrix evaluateIdentity(const Matrix& values);
        static Matrix evaluateIdentityDerivative(const Matrix& values, const Matrix& normalizedValues);

        static Matrix evaluateSoftmax(const Matrix& values);
        static Matrix evaluateSoftmaxDerivative(const Matrix& values, const Matrix& normalizedValues);
};

Matrix evaluateNormalization(NormalizationFunction lossFunction, const Matrix& values);
Matrix evaluateNormalizationDerivative(NormalizationFunction lossFunction, const Matrix& values, const Matrix& normalizedValues);

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
        static Matrix evaluateMeanSquaredErrorDerivative(const Matrix& predictedValues, const Matrix& expectedValues);

        static float evaluateSumSquaredError(const Matrix& predictedValues, const Matrix& expectedValues);
        static Matrix evaluateSumSquaredErrorDerivative(const Matrix& predictedValues, const Matrix& expectedValues);

        static float evaluateBinaryCrossEntropy(const Matrix& predictedValues, const Matrix& expectedValues);
        static Matrix evaluateBinaryCrossEntropyDerivative(const Matrix& predictedValues, const Matrix& expectedValues);

        static float evaluateCategoricalCrossEntropy(const Matrix& predictedValues, const Matrix& expectedValues);
        static Matrix evaluateCategoricalCrossEntropyDerivative(const Matrix& predictedValues, const Matrix& expectedValues);
};

float evaluateLoss(LossFunction lossFunction, const Matrix& predictedValues, const Matrix& expectedValues);
Matrix evaluateLossDerivative(LossFunction lossFunction, const Matrix& predictedValues, const Matrix& expectedValues);

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

        HiddenLayerParameters() = default;
        HiddenLayerParameters(int nodeCount, UnaryActivationFunction unaryActivationFunction);
        HiddenLayerParameters(UnaryActivationFunction unaryActivationFunction, const Matrix& weights, const Matrix& bias);

        // template <class Archive>
        // void serialize(Archive& ar) {
        //     ar(this->nodeCount, this->unaryActivationFunction, this->weights, this->bias);
        // };

        static constexpr float defaultMinInitialWeight = -0.1;
        static constexpr float defaultMaxInitialWeight = 0.1;

        static constexpr float defaultMinInitialBias = -0.1;
        static constexpr float defaultMaxInitialBias = 0.1;
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
        float loss;
        Matrix inputLayerLossPartials;
        std::vector<HiddenLayerLossPartials> hiddenLayersLossPartials;

        NetworkLossPartials() = default;
        NetworkLossPartials(float loss, const Matrix& inputLayerLossPartials, const std::vector<HiddenLayerLossPartials>& hiddenLayersLossPartials): loss(loss), inputLayerLossPartials(inputLayerLossPartials), hiddenLayersLossPartials(hiddenLayersLossPartials) {};

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
        LossFunction lossFunction;
        Matrix normalizedOutput;

        void runHiddenLayerFeedForward(int hiddenLayerIndex, const Matrix& input);

        // should only be called after feed forward has run
        void calculateHiddenLayerLossPartials(int hiddenLayerIndex, const Matrix& dLossWrtActivated);
    
    public:
        NeuralNetwork() = default;
        NeuralNetwork(int inputLayerNodeCount, std::vector<HiddenLayerParameters> hiddenLayerParameters, NormalizationFunction outputNormalizationFunction, LossFunction lossFunction);

        int getInputLayerNodeCount();
        std::vector<HiddenLayerState> getHiddenLayerStates();
        std::vector<HiddenLayerParameters> getHiddenLayerParameters();
        Matrix getNormalizedOutput();
        LossFunction getLossFunction();

        void initializeRandomHiddenLayerParameters();
        void initializeRandomHiddenLayerParameters(float minInitialWeight, float maxInitialWeight, float minInitialBias, float maxInitialBias);

        Matrix calculateFeedForwardOutput(const Matrix& input);

        float calculateLoss(const Matrix& expectedOutput); // must be called only after feedforward
        float calculateLoss(const Matrix& input, const Matrix& expectedOutput);

        NetworkLossPartials calculateLossPartials(const Matrix& expectedOutput); // must be called only after feedforward
        
        NetworkLossPartials calculateLossPartials(DataPoint dataPoint);
        NetworkLossPartials calculateBatchLossPartials(std::vector<DataPoint> dataBatch);

        void applyLossPartials(NetworkLossPartials lossPartial);

        void train(DataPoint trainingDataPoint, float learningRate);
        void batchTrain(std::vector<DataPoint> trainingDataBatch, float learningRate);

        std::string toString() const;

        // template <class Archive>
        // void serialize(Archive& ar) {
        //     ar(this->inputLayerNodeCount, this->hiddenLayerParameters, this->outputNormalizationFunction, this->lossFunction);
        // };
};

#endif