#include "curve_fitting.hpp"

void curveFittingExample()
{
    std::cout << "Curve Fitting Example" << std::endl << std::endl;

    NeuralNetwork nn(
        4,
        {
            HiddenLayerParameters(5, RELU),
            HiddenLayerParameters(8, RELU)
        },
        NormalizationFunction::IDENTITY,
        LossFunction::SUM_SQUARED_ERROR
    );

    std::cout << "Initial Neural Network" << std::endl;
    std::cout << nn.toString() << std::endl << std::endl;

    nn.calculateFeedForwardOutput(Matrix({{ 0.5 }, { -0.3 }, { 0.6 }, { 0.4 }}));

    std::cout << "After FF" << std::endl;
    std::cout << nn.toString() << std::endl << std::endl;
};