#include <iostream>

#include "./examples/curve_fitting.hpp"

int main()
{
    std::cout << "Approximating sin(x)" << std::endl;

    polynomialFitParametricCurve(
        "sin(x)",
        std::function<float(float)>([](float x) -> float { return sin(x); }),
        -2.0,
        2.0,
        100,
        200,
        NeuralNetwork(
            8,
            {
                HiddenLayerParameters(1, LINEAR)
            },
            IDENTITY,
            MEAN_SQUARED_ERROR
        ),
        0.1
    );

    std::cout << "Approximating cos(x)" << std::endl;

    polynomialFitParametricCurve(
        "cos(x)",
        std::function<float(float)>([](float x) -> float { return cos(x); }),
        -2.0,
        2.0,
        100,
        200,
        NeuralNetwork(
            9,
            {
                HiddenLayerParameters(1, LINEAR)
            },
            IDENTITY,
            MEAN_SQUARED_ERROR
        ),
        0.1
    );

    std::cout << "Approximating circle" << std::endl;

    nonlinearFitParametricCurve(
        "circle",
        std::function<std::pair<float, float>(float)>([](float t) -> std::pair<float, float> { return { sin(t), cos(t) }; }),
        0.0,
        2 * M_PI,
        50,
        200,
        NeuralNetwork(
            1,
            {
                HiddenLayerParameters(12, TANH),
                HiddenLayerParameters(12, TANH),
                HiddenLayerParameters(2, LINEAR)
            },
            IDENTITY,
            SUM_SQUARED_ERROR
        ),
        0.1
    );

    std::cout << "Approximating star" << std::endl;

    nonlinearFitParametricCurve(
        "star",
        std::function<std::pair<float, float>(float)>([](float t) -> std::pair<float, float> {
            auto r = 2 - 0.75 * sin(5.0 * t);

            auto x = r * cos(t);
            auto y = r * sin(t);

            return { x, y };
        }),
        0.0,
        2 * M_PI,
        50,
        300,
        NeuralNetwork(
            1,
            {
                HiddenLayerParameters(16, TANH),
                HiddenLayerParameters(16, TANH),
                HiddenLayerParameters(16, TANH),
                HiddenLayerParameters(2, LINEAR)
            },
            IDENTITY,
            SUM_SQUARED_ERROR
        ),
        0.1
    );
};