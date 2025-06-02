#ifndef EXAMPLES_CLASSIFICATION_HPP
#define EXAMPLES_CLASSIFICATION_HPP

#include "../../src/neural_network.hpp"

std::vector<std::vector<std::string>> parseSimpleCSV(std::string csvFilePath);

void benchmarkClassification(std::string dataSetName, std::vector<DataPoint> dataBatch, int trainingDataBatchSize, int sampleDataBatchSize, NeuralNetwork nn, float learningRate, int totalTrainingCycles, int trainingCyclesPerStep);

void benchmark2DClassification(std::string dataSetName, std::vector<DataPoint> dataBatch, int trainingDataBatchSize, int sampleDataBatchSize, NeuralNetwork nn, float learningRate, int totalTrainingCycles, int trainingCyclesPerStep, int trainingCyclesPer2DPlot);

#endif