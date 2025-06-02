#include <utility>
#include <sstream>

#include "../../lib/gnuplot-iostream.h"

#include "curve_fitting.hpp"

void plotParametricCurveWithPoints(
    std::string outputFilePath,
    std::string plotTitle,
    std::string labelText,
    std::function<std::pair<float, float>(float)> parametricCurve,
    float minInput,
    float maxInput,
    std::vector<std::pair<float, float>> points
)
{
    std::filesystem::path filePath(outputFilePath);
    if (!filePath.parent_path().empty()) std::filesystem::create_directories(filePath.parent_path());

    Gnuplot gp;

    gp << "set terminal pngcairo size 800,600 enhanced font 'Arial,12'\n";
    gp << "set output '" << outputFilePath << "'\n";
    gp << "set title '" << plotTitle << "'\n";
    gp << "set xlabel 'x'\n";
    gp << "set ylabel 'y'\n";
    gp << "set grid\n";

    if (!labelText.empty()) gp << "set label '" << labelText << "' at graph 0.02, 0.9 front\n";

    std::vector<std::pair<float, float>> curve;
    for (float t = minInput; t < maxInput; t += (maxInput - minInput) / 400.0f) {
        curve.emplace_back(parametricCurve(t));
    }

    gp << "plot '-' with lines lw 2 notitle, '-' with points pt 7 ps 0.5 lc rgb 'red' notitle\n";
    gp.send1d(curve);
    gp.send1d(points);
}

void plotParametricCurveWithPoints(
    std::string outputFilePath,
    std::string plotTitle,
    std::string labelText,
    std::function<float(float)> parametricCurve,
    float minInput,
    float maxInput,
    std::vector<std::pair<float, float>> points
)
{
    std::function<std::pair<float, float>(float)> reparameterizedCurve = [parametricCurve](float t) -> std::pair<float, float> { return { t, parametricCurve(t) }; };

    plotParametricCurveWithPoints(outputFilePath, plotTitle, labelText, reparameterizedCurve, minInput, maxInput, points);
};

float normalizeInput(float input, float minInput, float maxInput)
{
    return 2.0 * (input - minInput) / (maxInput - minInput) - 1.0;
};

float denormalizeInput(float normalizedInput, float minInput, float maxInput)
{
    return 0.5 * (normalizedInput + 1.0) * (maxInput - minInput) + minInput;
};

void polynomialFitParametricCurve(std::string curveName, std::function<float(float)> parametricCurve, float minInput, float maxInput, int trainingDataBatchSize, int sampleDataBatchSize, NeuralNetwork nn, float learningRate)
{
    if (minInput >= maxInput) throw std::runtime_error("polynomialFitParametricCurve: minInput must be less than maxInput");
    if (trainingDataBatchSize < 1) throw std::runtime_error("polynomialFitParametricCurve: trainingDataBatchSize must be at least 1");

    auto polynomialDegree = nn.getInputLayerNodeCount() - 1;
    if (polynomialDegree < 1) throw std::runtime_error("polynomialFitParametricCurve: polynomialDegree must be at least 1");

    std::vector<DataPoint> trainingDataBatch;
    for (int i = 0;i<trainingDataBatchSize;i++) {
        auto realX = minInput + (maxInput - minInput) * ((float) i / trainingDataBatchSize);
        auto x = normalizeInput(realX, minInput, maxInput);

        std::vector<float> xPowers = { 1.0 };
        for (int p = 0;p<polynomialDegree;p++) xPowers.push_back(x * xPowers.back());

        auto input = Matrix::transpose(Matrix({ xPowers }));

        Matrix expectedOutput({{ parametricCurve(realX) }});

        trainingDataBatch.push_back(DataPoint(input, expectedOutput));
    }

    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_real_distribution<float> sampleDataDistribution(minInput, maxInput);

    std::vector<DataPoint> sampleDataBatch;
    for (int i = 0;i<sampleDataBatchSize;i++) {
        auto realX = sampleDataDistribution(rng);
        auto x = normalizeInput(realX, minInput, maxInput);

        std::vector<float> xPowers = { 1.0 };
        for (int p = 0;p<polynomialDegree;p++) xPowers.push_back(x * xPowers.back());

        auto input = Matrix::transpose(Matrix({ xPowers }));

        Matrix expectedOutput({{ parametricCurve(realX) }});

        sampleDataBatch.push_back(DataPoint(input, expectedOutput));
    }

    nn.initializeRandomLayerParameters(-0.5, 0.5, -0.5, 0.5);

    for (int i = 0;i<=5000;i++) {
        if (i % 1000 == 0) {
            std::string cycleCount = std::to_string(i);

            auto netLoss = 0.0;
            std::vector<std::pair<float, float>> points;
            
            for (auto sampleDataPoint : sampleDataBatch) {
                auto normalizedX = sampleDataPoint.input.get(1, 0); // x^1
                auto x = denormalizeInput(normalizedX, minInput, maxInput);

                netLoss += nn.calculateLoss(sampleDataPoint.input, sampleDataPoint.expectedOutput);

                auto y = nn.getNormalizedOutput().get(0, 0);

                points.emplace_back(x, y);
            }

            plotParametricCurveWithPoints(
                "./results/polynomialFitParametricCurve/" + curveName + "/after" + cycleCount + ".png",
                "Degree " + std::to_string(polynomialDegree) + " Polynomial Approximation of " + curveName + " After " + cycleCount + " Training Cycles",
                "Average Error Squared: " + std::to_string(netLoss / sampleDataBatch.size()),
                parametricCurve,
                minInput,
                maxInput,
                points
            );
        }

        nn.batchTrain(trainingDataBatch, learningRate);
    }
    
    auto learnedParameters = nn.getHiddenLayerParameters()[0];

    std::cout << curveName << " ~ " << (learnedParameters.bias.get(0, 0) + learnedParameters.weights.get(0, 0));

    // the learned weights act in the normalized space, 2.0 since normalization maps to (-1.0, 1.0)
    float weightScaleFactor = 2.0 / (maxInput - minInput);

    for (int i = 0;i<polynomialDegree;i++) {
        auto denormalizedWeight = learnedParameters.weights.get(0, i + 1) * pow(weightScaleFactor, i + 1);

        std::cout << " + x^" << (i + 1) << " * " << denormalizedWeight;
    }

    std::cout << std::endl;
};

void nonlinearFitParametricCurve(std::string curveName, std::function<std::pair<float, float>(float)> parametricCurve, float minInput, float maxInput, int trainingDataBatchSize, int sampleDataBatchSize, NeuralNetwork nn, float learningRate)
{
    if (minInput >= maxInput) throw std::runtime_error("polynomialFitParametricCurve: minInput must be less than maxInput");
    if (trainingDataBatchSize < 1) throw std::runtime_error("polynomialFitParametricCurve: trainingDataBatchSize must be at least 1");

    std::vector<DataPoint> trainingDataBatch;
    for (int i = 0;i<trainingDataBatchSize;i++) {
        auto realT = minInput + (maxInput - minInput) * ((float) i / trainingDataBatchSize);
        auto t = normalizeInput(realT, minInput, maxInput);

        Matrix input({{ t }});

        auto [expectedOutputX, expectedOutputY] = parametricCurve(realT);

        Matrix expectedOutput(std::vector<std::vector<float>>({{ expectedOutputX }, { expectedOutputY }}));

        trainingDataBatch.push_back(DataPoint(input, expectedOutput));
    }

    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_real_distribution<float> sampleDataDistribution(minInput, maxInput);

    std::vector<DataPoint> sampleDataBatch;
    for (int i = 0;i<sampleDataBatchSize;i++) {
        auto realT = sampleDataDistribution(rng);
        auto t = normalizeInput(realT, minInput, maxInput);
        
        Matrix input({{ t }});

        auto [expectedOutputX, expectedOutputY] = parametricCurve(realT);

        Matrix expectedOutput(std::vector<std::vector<float>>({{ expectedOutputX }, { expectedOutputY }}));

        sampleDataBatch.push_back(DataPoint(input, expectedOutput));
    }

    nn.initializeRandomLayerParameters(-0.5, 0.5, -0.5, 0.5);

    for (int i = 0;i<=5000;i++) {
        if (i % 1000 == 0) {
            std::string cycleCount = std::to_string(i);

            auto netLoss = 0.0;
            std::vector<std::pair<float, float>> points;
            
            for (auto sampleDataPoint : sampleDataBatch) {
                netLoss += nn.calculateLoss(sampleDataPoint.input, sampleDataPoint.expectedOutput);

                auto observedOutput = nn.getNormalizedOutput();

                points.emplace_back(observedOutput.get(0, 0), observedOutput.get(1, 0));
            }

            plotParametricCurveWithPoints(
                "./results/nonlinearFitParametricCurve/" + curveName + "/after" + cycleCount + ".png",
                "Nonlinear Approximation of " + curveName + " After " + cycleCount + " Training Cycles",
                "Average Error Squared: " + std::to_string(netLoss / sampleDataBatch.size()),
                parametricCurve,
                minInput,
                maxInput,
                points
            );
        }

        nn.batchTrain(trainingDataBatch, learningRate);
    }
};