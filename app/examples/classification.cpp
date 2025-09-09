#include <fstream>
#include <filesystem>
#include <utility>
#include <random>

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

std::string getColor(int index) {
    std::vector<std::string> colors = { "red", "blue", "green", "orange", "purple", "cyan", "magenta", "brown", "black" };

    return colors[index % colors.size()];
}

void plot2DClassification(std::string outputFilePath, std::string plotTitle, std::vector<std::vector<std::pair<float, float>>> classifiedPoints)
{
    std::filesystem::path filePath(outputFilePath);
    if (!filePath.parent_path().empty()) std::filesystem::create_directories(filePath.parent_path());

    Gnuplot gp;

    gp << "set terminal pngcairo size 1000,600 enhanced font 'Arial,12'\n";
    gp << "set output '" << outputFilePath << "'\n";

    gp << "set title '" << plotTitle << "'\n";
    gp << "set xlabel 'X'\n";
    gp << "set ylabel 'Y'\n";
    gp << "set grid\n";

    gp << "plot ";

    for (int i = 0;i<classifiedPoints.size();i++) {
        if (i > 0) gp << ", ";

        gp << "'-' with points pointtype 7 pointsize 0.5 lc rgb '" << getColor(i) << "' notitle";
    }
    gp << "\n";

    for (auto pointsClass : classifiedPoints) gp.send1d(pointsClass);
};

void benchmarkClassification(std::string dataSetName, std::vector<DataPoint> dataBatch, int trainingDataBatchSize, int sampleDataBatchSize, NeuralNetwork nn, float learningRate, int totalTrainingCycles, int trainingCyclesPerStep)
{
    nn.initializeRandomHiddenLayerParameters(-0.5, 0.5, -0.5, 0.5);

    std::vector<float> percentAccuratelyClassified;
    std::vector<float> averageLoss;

    std::random_device rd;
    std::mt19937 rng(rd());

    for (int i = 0;i<=totalTrainingCycles;i++) {
        std::shuffle(dataBatch.begin(), dataBatch.end(), rng);
        std::vector<DataPoint> trainingDataBatch(dataBatch.begin(), dataBatch.begin() + trainingDataBatchSize);
        std::vector<DataPoint> sampleDataBatch(dataBatch.end() - sampleDataBatchSize, dataBatch.end());

        if (i % trainingCyclesPerStep == 0) {
            float netLoss = 0.0;
            float samplesAccuratelyClassified = 0.0;

            for (auto sampleDataPoint : sampleDataBatch) {
                netLoss += nn.calculateLoss(sampleDataPoint.input, sampleDataPoint.expectedOutput);

                auto expectedOutput = sampleDataPoint.expectedOutput;
                auto observedOutput = nn.getNormalizedOutput();

                auto expectedClassificationIndex = 0;
                auto observedClassificationIndex = 0;

                for (int r = 0;r<expectedOutput.shape()[0];r++) {
                    if (expectedOutput(r, 0) > expectedOutput(expectedClassificationIndex, 0)) expectedClassificationIndex = r;
                    if (observedOutput(r, 0) > observedOutput(observedClassificationIndex, 0)) observedClassificationIndex = r;
                }
                
                if (expectedClassificationIndex == observedClassificationIndex) samplesAccuratelyClassified++;
            }

            percentAccuratelyClassified.push_back(100.0 * samplesAccuratelyClassified / sampleDataBatch.size());
            averageLoss.push_back(netLoss / sampleDataBatch.size());
        }

        nn.batchTrain(trainingDataBatch, learningRate);
    }

    plotClassificationAccuracy(
        "./results/classification/" + dataSetName + "/benchmark.png",
        dataSetName + " Classification Benchmark",
        trainingCyclesPerStep,
        percentAccuratelyClassified,
        averageLoss
    );
};

void benchmark2DClassification(std::string dataSetName, std::vector<DataPoint> dataBatch, int trainingDataBatchSize, int sampleDataBatchSize, NeuralNetwork nn, float learningRate, int totalTrainingCycles, int trainingCyclesPerStep, int trainingCyclesPer2DPlot)
{
    nn.initializeRandomHiddenLayerParameters(-0.5, 0.5, -0.5, 0.5);

    std::vector<float> percentAccuratelyClassified;
    std::vector<float> averageLoss;

    std::random_device rd;
    std::mt19937 rng(rd());

    for (int i = 0;i<=totalTrainingCycles;i++) {
        std::shuffle(dataBatch.begin(), dataBatch.end(), rng);
        std::vector<DataPoint> trainingDataBatch(dataBatch.begin(), dataBatch.begin() + trainingDataBatchSize);
        std::vector<DataPoint> sampleDataBatch(dataBatch.end() - sampleDataBatchSize, dataBatch.end());

        if (i % trainingCyclesPerStep == 0) {
            float netLoss = 0.0;
            float samplesAccuratelyClassified = 0.0;

            for (auto sampleDataPoint : sampleDataBatch) {
                netLoss += nn.calculateLoss(sampleDataPoint.input, sampleDataPoint.expectedOutput);

                auto expectedOutput = sampleDataPoint.expectedOutput;
                auto observedOutput = nn.getNormalizedOutput();

                auto expectedClassificationIndex = 0;
                auto observedClassificationIndex = 0;

                for (int r = 0;r<expectedOutput.shape()[0];r++) {
                    if (expectedOutput(r, 0) > expectedOutput(expectedClassificationIndex, 0)) expectedClassificationIndex = r;
                    if (observedOutput(r, 0) > observedOutput(observedClassificationIndex, 0)) observedClassificationIndex = r;
                }
                
                if (expectedClassificationIndex == observedClassificationIndex) samplesAccuratelyClassified++;
            }

            percentAccuratelyClassified.push_back(100.0 * samplesAccuratelyClassified / sampleDataBatch.size());
            averageLoss.push_back(netLoss / sampleDataBatch.size());
        }

        if (i % trainingCyclesPer2DPlot == 0) {
            std::vector<std::vector<std::pair<float, float>>> classifiedPoints;

            classifiedPoints.resize(nn.getNormalizedOutput().shape()[0]);

            for (auto dataPoint : dataBatch) {
                auto pointClassificationOutput = nn.calculateFeedForwardOutput(dataPoint.input);

                int classificationIndex = 0;

                for (int r = 0;r<pointClassificationOutput.shape()[0];r++) {
                    if (pointClassificationOutput(r, 0) > pointClassificationOutput(classificationIndex, 0)) classificationIndex = r;
                }

                classifiedPoints[classificationIndex].emplace_back(dataPoint.input(0, 0), dataPoint.input(1, 0));
            }
            
            std::vector<std::vector<std::pair<float, float>>> nonEmptyClasses;

            for (auto pointsClass : classifiedPoints) if (!pointsClass.empty()) nonEmptyClasses.push_back(pointsClass);

            plot2DClassification(
                "./results/classification2D/" + dataSetName + "/after" + std::to_string(i) + ".png",
                dataSetName + " Classifications After " + std::to_string(i) + " Training Cycles",
                nonEmptyClasses
            );
        }

        nn.batchTrain(trainingDataBatch, learningRate);
    }

    plotClassificationAccuracy(
        "./results/classification2d/" + dataSetName + "/benchmark.png",
        dataSetName + " Classification Benchmark",
        trainingCyclesPerStep,
        percentAccuratelyClassified,
        averageLoss
    );
};