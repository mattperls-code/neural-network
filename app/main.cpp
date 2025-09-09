#include <iostream>
#include <algorithm>
#include <chrono>

#include "./examples/classification.hpp"
#include "./examples/curve_fitting.hpp"

int main()
{
    auto start = std::chrono::high_resolution_clock::now();

    std::cout << "Classifying iris" << std::endl;

    auto irisCsvData = parseSimpleCSV("./app/examples/data/iris.csv");
    irisCsvData.erase(irisCsvData.begin());

    std::vector<DataPoint> irisDataBatch;

    for (int i = 0;i<irisCsvData.size();i++) {
        Matrix input = {
            { std::stof(irisCsvData[i][0]) },
            { std::stof(irisCsvData[i][1]) },
            { std::stof(irisCsvData[i][2]) },
            { std::stof(irisCsvData[i][3]) }
        };
        
        Matrix expectedOutput = {
            { irisCsvData[i][4] == "Iris-setosa" ? 1.0f : 0.0f },
            { irisCsvData[i][4] == "Iris-versicolor" ? 1.0f : 0.0f },
            { irisCsvData[i][4] == "Iris-virginica" ? 1.0f : 0.0f }
        };

        irisDataBatch.emplace_back(input, expectedOutput);
    }
    
    benchmarkClassification(
        "iris",
        irisDataBatch,
        100,
        50,
        NeuralNetwork(
            4,
            {
                HiddenLayerParameters(12, RELU),
                HiddenLayerParameters(12, RELU),
                HiddenLayerParameters(3, LINEAR)
            },
            NormalizationFunction::SOFTMAX,
            CATEGORICAL_CROSS_ENTROPY
        ),
        0.01,
        3000,
        10
    );

    std::cout << "Classifying regions" << std::endl;
    
    std::vector<DataPoint> regionsDataBatch;

    for (float x = -3.0;x<=3.0;x+=0.05) {
        for (float y = -3.0;y<=3.0;y+=0.05) {
            Matrix input = {{ x }, { y }};

            auto r = sqrt(x * x + y * y);
            auto theta = atan2(y, x);

            auto outerBoundary = 2 - cos(3.0 * theta) * sin(3.0 * theta);
            auto innerBoundary = 0.5 * outerBoundary;
            
            float isOutside = r > outerBoundary ? 1.0 : 0.0;
            float isInside = r < innerBoundary ? 1.0 : 0.0;
            float isInbetween = 1.0 - isOutside - isInside;

            Matrix expectedOutput({{ isOutside }, { isInside }, { isInbetween }});

            regionsDataBatch.emplace_back(input, expectedOutput);
        }
    }

    benchmark2DClassification(
        "regions",
        regionsDataBatch,
        200,
        500,
        NeuralNetwork(
            2,
            {
                HiddenLayerParameters(12, TANH),
                HiddenLayerParameters(12, TANH),
                HiddenLayerParameters(3, LINEAR)
            },
            SOFTMAX,
            CATEGORICAL_CROSS_ENTROPY
        ),
        0.25,
        3000,
        10,
        500
    );

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

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;

    std::cout << "Execution time: " << elapsed.count() << " seconds\n";

    return 0;
};