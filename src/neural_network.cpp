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