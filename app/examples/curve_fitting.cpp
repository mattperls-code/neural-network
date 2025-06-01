#include "curve_fitting.hpp"

void polynomialFit1DCurve(std::string curveName, std::function<float(float)> curveParametric, float minInput, float maxInput, int numDataPoints, int polynomialDegree)
{
    if (minInput >= maxInput) throw std::runtime_error("polynomialFit1DCurve: minInput must be less than maxInput");
    if (numDataPoints < 1) throw std::runtime_error("polynomialFit1DCurve: numDataPoints must be at least 1");
    if (polynomialDegree < 0) throw std::runtime_error("polynomialFit1DCurve: polynomialDegree must be non-negative");

    std::vector<DataPoint> trainingDataBatch;
    for (int i = 0;i<numDataPoints;i++) {
        auto x = minInput + (maxInput - minInput) * ((float) i / (float) numDataPoints);

        std::vector<float> xPowers = { 1.0 };
        for (int p = 0;p<polynomialDegree;p++) xPowers.push_back(x * xPowers.back());

        auto input = Matrix::transpose(Matrix({ xPowers }));

        Matrix expectedOutput({{ curveParametric(x) }});

        trainingDataBatch.push_back(DataPoint(input, expectedOutput));
    }

    NeuralNetwork nn(
        polynomialDegree + 1,
        {
            HiddenLayerParameters(1, LINEAR)
        },
        IDENTITY,
        MEAN_SQUARED_ERROR
    );

    nn.initializeRandomLayerParameters(-0.5, 0.5, -0.5, 0.5);

    for (int i = 0;i<10000;i++) {
        // std::cout << "Training Cycle " << i << std::endl;

        nn.batchTrain(trainingDataBatch, 0.005);
    }
    
    auto learnedParameters = nn.getHiddenLayerParameters()[0];

    std::cout << "Weights: " << std::endl;
    std::cout << learnedParameters.weights.toString() << std::endl << std::endl;

    std::cout << "Bias: " << std::endl;
    std::cout << learnedParameters.bias.toString() << std::endl << std::endl;
};