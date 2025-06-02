#ifndef EXAMPLES_CLASSIFICATION_HPP
#define EXAMPLES_CLASSIFICATION_HPP

#include "../../src/neural_network.hpp"

std::vector<std::vector<std::string>> parseSimpleCSV(std::string csvFilePath);

void benchmarkClassification(std::string dataSetName, std::vector<DataPoint> trainingDataBatch, std::vector<DataPoint> sampleDataBatch, NeuralNetwork nn, float learningRate, int totalTrainingCycles, int trainingCyclesPerStep);

#endif