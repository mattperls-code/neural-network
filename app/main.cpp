#include <iostream>
#include <algorithm>

#include "./examples/classification.hpp"
#include "./examples/curve_fitting.hpp"

int main()
{
    std::cout << "Classifying iris" << std::endl;

    auto irisCsvData = parseSimpleCSV("./app/examples/data/iris.csv");
    irisCsvData.erase(irisCsvData.begin());
    
    std::random_device rd;
    std::mt19937 rng(rd());
    std::shuffle(irisCsvData.begin(), irisCsvData.end(), rng);

    std::vector<DataPoint> irisTrainingDataBatch;
    std::vector<DataPoint> irisSampleDataBatch;

    for (int i = 0;i<irisCsvData.size();i++) {
        Matrix input(Shape(4, 1));
        
        for (int j = 0;j<4;j++) input.set(j, 0, std::stof(irisCsvData[i][j]));

        Matrix expectedOutput(Shape(3, 1));

        expectedOutput.set(0, 0, irisCsvData[i][4] == "Iris-setosa" ? 1.0 : 0.0);
        expectedOutput.set(1, 0, irisCsvData[i][4] == "Iris-versicolor" ? 1.0 : 0.0);
        expectedOutput.set(2, 0, irisCsvData[i][4] == "Iris-virginica" ? 1.0 : 0.0);

        auto& dataBatch = (i < 0.5 * irisCsvData.size()) ? irisTrainingDataBatch : irisSampleDataBatch;

        dataBatch.emplace_back(input, expectedOutput);
    }
    
    benchmarkClassification(
        "iris",
        irisTrainingDataBatch,
        irisSampleDataBatch,
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
        1000,
        5
    );

    std::cout << "Classifying wine" << std::endl;

    auto wineCsvData = parseSimpleCSV("./app/examples/data/wine.csv");
    wineCsvData.erase(wineCsvData.begin());

    std::vector<DataPoint> wineTrainingDataBatch;
    std::vector<DataPoint> wineSampleDataBatch;

    for (int i = 0;i<wineCsvData.size();i++) {
        Matrix input(Shape(4, 1));
        
        for (int j = 0;j<4;j++) input.set(j, 0, std::stof(wineCsvData[i][j]));

        Matrix expectedOutput(Shape(3, 1));

        expectedOutput.set(0, 0, wineCsvData[i][4] == "low" ? 1.0 : 0.0);
        expectedOutput.set(1, 0, wineCsvData[i][4] == "medium" ? 1.0 : 0.0);
        expectedOutput.set(2, 0, wineCsvData[i][4] == "high" ? 1.0 : 0.0);

        auto& dataBatch = (i < 0.35 * wineCsvData.size()) ? wineTrainingDataBatch : wineSampleDataBatch;

        dataBatch.emplace_back(input, expectedOutput);
    }
    
    benchmarkClassification(
        "wine",
        wineTrainingDataBatch,
        wineSampleDataBatch,
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
        0.05,
        4000,
        50
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
};