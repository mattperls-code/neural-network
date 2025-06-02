#ifndef EXAMPLES_CURVE_FITTING_HPP
#define EXAMPLES_CURVE_FITTING_HPP

#include <functional>

#include "../../src/neural_network.hpp"

void polynomialFitParametricCurve(std::string curveName, std::function<float(float)> parametricCurve, float minInput, float maxInput, int trainingDataBatchSize, int sampleDataBatchSize, NeuralNetwork nn, float learningRate);
void nonlinearFitParametricCurve(std::string curveName, std::function<std::pair<float, float>(float)> parametricCurve, float minInput, float maxInput, int trainingDataBatchSize, int sampleDataBatchSize, NeuralNetwork nn, float learningRate);

#endif