#include <fstream>
#include <filesystem>
#include <utility>

#include "../../lib/gnuplot-iostream.h"

#include "classification.hpp"

std::vector<std::vector<std::string>> parseSimpleCSV(std::string csvFilePath)
{
    std::vector<std::vector<std::string>> output;

    std::ifstream csvFile(csvFilePath);

    if (!csvFile.good()) throw std::runtime_error("parseCSV: failed to open csvFilePath");

    std::string line;

    while (getline(csvFile, line))
    {
        std::vector<std::string> lineData = { "" };

        while (!line.empty())
        {
            if (line[0] == ',') lineData.push_back("");
            else if (line[0] != '\r') lineData.back() += line[0];

            line.erase(line.begin());
        };

        output.push_back(lineData);
    };

    return output;
};

void plotClassificationAccuracy(std::string outputFilePath, std::string plotTitle, int trainingCyclesPerStep, std::vector<float> percentAccuratelyClassified, std::vector<float> averageLoss)
{
    if (percentAccuratelyClassified.size() != averageLoss.size()) throw std::runtime_error("plotClassificationAccuracy: mismatched input sizes");

    std::filesystem::path filePath(outputFilePath);
    if (!filePath.parent_path().empty()) std::filesystem::create_directories(filePath.parent_path());

    Gnuplot gp;

    gp << "set terminal pngcairo size 1000,600 enhanced font 'Arial,12'\n";
    gp << "set output '" << outputFilePath << "'\n";

    gp << "set title '" << plotTitle << "'\n";
    gp << "set xlabel 'Training Cycles'\n";
    gp << "set grid\n";

    gp << "set ylabel 'Accuracy (%)' textcolor rgb 'blue'\n";
    gp << "set yrange [0:100]\n";
    gp << "set y2label 'Average Error' textcolor rgb 'red'\n";
    gp << "set y2range [0:*]\n";
    gp << "set y2tics\n";

    gp << "set key off\n";
    gp << "set rmargin 10\n";

    gp << "plot '-' using 1:2 with lines lw 2 lc rgb 'blue' title 'Accuracy (%)', "
          "'-' using 1:2 axes x1y2 with lines lw 2 lc rgb 'red' title 'Average Error'\n";

    std::vector<std::pair<int, float>> accuracyPoints;
    std::vector<std::pair<int, float>> errorPoints;

    for (int i = 0;i<percentAccuratelyClassified.size();i++) {
        accuracyPoints.emplace_back(i * trainingCyclesPerStep, percentAccuratelyClassified[i]);
        errorPoints.emplace_back(i * trainingCyclesPerStep, averageLoss[i]);
    }

    gp.send1d(accuracyPoints);
    gp.send1d(errorPoints);
};

void benchmarkClassification(std::string dataSetName, std::vector<DataPoint> trainingDataBatch, std::vector<DataPoint> sampleDataBatch, NeuralNetwork nn, float learningRate, int totalTrainingCycles, int trainingCyclesPerStep)
{
    nn.initializeRandomLayerParameters(-0.5, 0.5, -0.5, 0.5);

    std::vector<float> percentAccuratelyClassified;
    std::vector<float> averageLoss;

    for (int i = 0;i<=totalTrainingCycles;i++) {
        if (i % trainingCyclesPerStep == 0) {
            float netLoss = 0.0;
            float samplesAccuratelyClassified = 0.0;

            for (auto sampleDataPoint : sampleDataBatch) {
                netLoss += nn.calculateLoss(sampleDataPoint.input, sampleDataPoint.expectedOutput);

                auto expectedOutput = sampleDataPoint.expectedOutput;
                auto observedOutput = nn.getNormalizedOutput();

                auto expectedClassificationIndex = 0;
                auto observedClassificationIndex = 0;

                for (int r = 0;r<expectedOutput.rowCount();r++) {
                    if (expectedOutput.get(r, 0) > expectedOutput.get(expectedClassificationIndex, 0)) expectedClassificationIndex = r;
                    if (observedOutput.get(r, 0) > observedOutput.get(observedClassificationIndex, 0)) observedClassificationIndex = r;
                }
                
                if (expectedClassificationIndex == observedClassificationIndex) samplesAccuratelyClassified++;
            }

            percentAccuratelyClassified.push_back(100.0 * samplesAccuratelyClassified / sampleDataBatch.size());
            averageLoss.push_back(netLoss / sampleDataBatch.size());
        }

        nn.batchTrain(trainingDataBatch, learningRate);
    }

    plotClassificationAccuracy(
        "./results/classification/" + dataSetName + ".png",
        dataSetName + " Classification Benchmark",
        trainingCyclesPerStep,
        percentAccuratelyClassified,
        averageLoss
    );
};