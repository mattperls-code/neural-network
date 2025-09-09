#include <catch2/catch_all.hpp>
#include <iostream>

#include "../src/neural_network.hpp"

bool matricesAreApproxEqual(const Matrix& matA, const Matrix& matB)
{
    if (!xt::all(abs(matA - matB) < 0.001)) {
        std::cout << "matA: " << matrixToStr(matA) << std::endl;
        std::cout << "matB: " << matrixToStr(matB) << std::endl;

        return false;
    }

    return true;
};

TEST_CASE("NEURAL NETWORK") {
    SECTION("UNARY ACTIVATION FUNCTION") {
        SECTION("LINEAR") {
            Matrix input({{ 1.0, -2.0 }, { 3.5, 0.0 }});
            Matrix expectedOutput = input;
            Matrix observedOutput = UnaryActivationFunctionImplementation::evaluateLinear(input);

            REQUIRE(matricesAreApproxEqual(observedOutput, expectedOutput));
    
            Matrix expectedDerivative(input.shape(), 1.0);
            Matrix observedDerivative = UnaryActivationFunctionImplementation::evaluateLinearDerivative(input, observedOutput);

            REQUIRE(matricesAreApproxEqual(observedDerivative, expectedDerivative));
        }
    
        SECTION("RELU") {
            Matrix input({{ 1.0, -2.0 }, { 0.0, 3.0 }});
            Matrix expectedOutput({{ 1.0, 0.0 }, { 0.0, 3.0 }});
            Matrix observedOutput = UnaryActivationFunctionImplementation::evaluateRelu(input);

            REQUIRE(matricesAreApproxEqual(observedOutput, expectedOutput));
    
            Matrix expectedDerivative({{ 1.0, 0.0 }, { 0.0, 1.0 }});
            Matrix observedDerivative = UnaryActivationFunctionImplementation::evaluateReluDerivative(input, observedOutput);

            REQUIRE(matricesAreApproxEqual(observedDerivative, expectedDerivative));
        }
    
        SECTION("SIGMOID") {
            Matrix input({{ 0.0 }});
            Matrix expectedOutput({{ 0.5 }});
            Matrix observedOutput = UnaryActivationFunctionImplementation::evaluateSigmoid(input);

            REQUIRE(matricesAreApproxEqual(observedOutput, expectedOutput));
    
            Matrix expectedDerivative({{ 0.25 }});
            Matrix observedDerivative = UnaryActivationFunctionImplementation::evaluateSigmoidDerivative(input, observedOutput);

            REQUIRE(matricesAreApproxEqual(observedDerivative, expectedDerivative));
        }
    
        SECTION("TANH") {
            Matrix input({{ 0.0 }});
            Matrix expectedOutput({{ 0.0 }});
            Matrix observedOutput = UnaryActivationFunctionImplementation::evaluateTanh(input);

            REQUIRE(matricesAreApproxEqual(observedOutput, expectedOutput));
    
            Matrix expectedDerivative({{ 1.0 }});
            Matrix observedDerivative = UnaryActivationFunctionImplementation::evaluateTanhDerivative(input, observedOutput);

            REQUIRE(matricesAreApproxEqual(observedDerivative, expectedDerivative));
        }
    
        SECTION("EVALUATE") {
            Matrix input = {{ 1.0, -1.0 }};

            REQUIRE(matricesAreApproxEqual(evaluateUnaryActivation(LINEAR, input), UnaryActivationFunctionImplementation::evaluateLinear(input)));
            REQUIRE(matricesAreApproxEqual(evaluateUnaryActivation(RELU, input), UnaryActivationFunctionImplementation::evaluateRelu(input)));
    
            auto sigmoid = evaluateUnaryActivation(SIGMOID, input);
            auto expectedSigmoid = UnaryActivationFunctionImplementation::evaluateSigmoid(input);

            REQUIRE(matricesAreApproxEqual(sigmoid, expectedSigmoid));
        }
    
        SECTION("DERIVATIVE") {
            Matrix input = {{ 0.5 }};
            Matrix activatedSigmoid = UnaryActivationFunctionImplementation::evaluateSigmoid(input);
            auto derivative = evaluateUnaryActivationDerivative(SIGMOID, input, activatedSigmoid);
            auto expected = UnaryActivationFunctionImplementation::evaluateSigmoidDerivative(input, activatedSigmoid);

            REQUIRE(matricesAreApproxEqual(derivative, expected));
        }
    }

    SECTION("NORMALIZATION FUNCTION") {
        SECTION("IDENTITY") {
            Matrix input = {{ 3.5 }, { -2.0 }, { 0.0 }};
            Matrix expectedOutput = input;
            Matrix observedOutput = NormalizationFunctionImplementation::evaluateIdentity(input);

            REQUIRE(matricesAreApproxEqual(observedOutput, expectedOutput));
    
            Matrix observedDerivative = NormalizationFunctionImplementation::evaluateIdentityDerivative(input, observedOutput);

            for (int i = 0;i<observedOutput.shape()[0];i++) {
                for (int j = 0;j<observedOutput.shape()[0];j++) {
                    auto expected = (i == j) ? 1.0 : 0.0;

                    REQUIRE(Catch::Approx(observedDerivative(i, j)).epsilon(0.1) == expected);
                }
            }
        }
    
        SECTION("SOFTMAX") {
            Matrix input = {{ 1.0 }, { 2.0 }, { 3.0 }};
            Matrix observedOutput = NormalizationFunctionImplementation::evaluateSoftmax(input);
    
            auto sum = 0.0;
            for (int r = 0;r<observedOutput.shape()[0];r++) sum += observedOutput(r, 0);
    
            REQUIRE(Catch::Approx(sum).epsilon(0.0001) == 1.0);
    
            Matrix observedDerivative = NormalizationFunctionImplementation::evaluateSoftmaxDerivative(input, observedOutput);

            for (int i = 0;i<observedOutput.shape()[0];i++) {
                for (int j = 0;j<observedOutput.shape()[0];j++) {
                    auto expected = (i == j) ? (observedOutput(i, 0) * (1.0 - observedOutput(i, 0))) : (-observedOutput(i, 0) * observedOutput(j, 0));
    
                    REQUIRE(Catch::Approx(observedDerivative(i, j)).epsilon(0.0001) == expected);
                }
            }
        }
    
        SECTION("EVALUATE AND DERIVATIVE - IDENTITY") {
            Matrix input = {{ 5.0 }, { -3.0 }};
            Matrix expectedOutput = input;
            Matrix observedOutput = evaluateNormalization(NormalizationFunction::IDENTITY, input);

            REQUIRE(matricesAreApproxEqual(observedOutput, expectedOutput));
    
            Matrix observedDerivative = evaluateNormalizationDerivative(NormalizationFunction::IDENTITY, input, observedOutput);

            for (int i = 0;i<observedOutput.shape()[0];i++) {
                for (int j = 0;j<observedOutput.shape()[0];j++) {
                    auto expected = (i == j) ? 1.0 : 0.0;

                    REQUIRE(Catch::Approx(observedDerivative(i, j)).epsilon(0.1) == expected);
                }
            }
        }
    
        SECTION("EVALUATE AND DERIVATIVE - SOFTMAX") {
            Matrix input = {{ 0.0 }, { 1.0 }, { 2.0 }};
            Matrix observedOutput = evaluateNormalization(NormalizationFunction::SOFTMAX, input);
    
            auto sum = 0.0;
            for (int r = 0;r<observedOutput.shape()[0];r++) sum += observedOutput(r, 0);
    
            REQUIRE(Catch::Approx(sum).epsilon(0.0001) == 1.0);
    
            Matrix observedDerivative = evaluateNormalizationDerivative(NormalizationFunction::SOFTMAX, input, observedOutput);

            for (int i = 0;i<observedOutput.shape()[0];i++) {
                for (int j = 0;j<observedOutput.shape()[0];j++) {
                    auto expected = (i == j) ? (observedOutput(i, 0) * (1.0 - observedOutput(i, 0))) : (-observedOutput(i, 0) * observedOutput(j, 0));
    
                    REQUIRE(Catch::Approx(observedDerivative(i, j)).epsilon(0.0001) == expected);
                }
            }
        }
    }

    SECTION("LOSS FUNCTION") {
        SECTION("EVALUATE MSE") {
            Matrix predicted = {{ 1.0 }, { 2.0 }, { 3.0 }};
            Matrix expected = {{ 1.0 }, { 2.0 }, { 2.0 }};
    
            auto expectedOutput = ((0.0 * 0.0) + (0.0 * 0.0) + (1.0 * 1.0)) / 3.0;
            auto observedOutput = LossFunctionImplementation::evaluateMeanSquaredError(predicted, expected);

            REQUIRE(observedOutput == Catch::Approx(expectedOutput));
    
            Matrix predicted2 = {{ 0.0 }, { 0.0 }};
            Matrix expected2 = {{ 1.0 }, { 1.0 }};
    
            auto expectedOutput2 = ((-1.0 * -1.0) + (-1.0 * -1.0)) / 2.0;
            observedOutput = LossFunctionImplementation::evaluateMeanSquaredError(predicted2, expected2);

            REQUIRE(observedOutput == Catch::Approx(expectedOutput2));
        }
    
        SECTION("MSE DERIVATIVE") {
            Matrix predicted = {{ 3.0 }, { 4.0 }};
            Matrix expected = {{ 1.0 }, { 2.0 }};
            Matrix observedOutput = LossFunctionImplementation::evaluateMeanSquaredErrorDerivative(predicted, expected);
    
            auto coeff = 2.0 / 2.0;
            for (int r = 0;r<2;r++) {
                auto expectedValue = coeff * (predicted(r, 0) - expected(r, 0));

                REQUIRE(observedOutput(r, 0) == Catch::Approx(expectedValue));
            }
    
            Matrix predicted2 = {{ -1.0 }, { 0.0 }, { 1.0 }};
            Matrix expected2 = {{ 0.0 }, { 0.0 }, { 0.0 }};
            observedOutput = LossFunctionImplementation::evaluateMeanSquaredErrorDerivative(predicted2, expected2);
    
            coeff = 2.0 / 3.0;
            for (int r = 0;r<3;r++) {
                auto expectedValue = coeff * (predicted2(r, 0) - expected2(r, 0));

                REQUIRE(observedOutput(r, 0) == Catch::Approx(expectedValue));
            }
        }
    
        SECTION("EVALUATE SSE") {
            Matrix predicted = {{ 1.0 }, { 3.0 }};
            Matrix expected = {{ 2.0 }, { 2.0 }};
            auto expectedOutput = (( -1.0 * -1.0 ) + ( 1.0 * 1.0 ));
            auto observedOutput = LossFunctionImplementation::evaluateSumSquaredError(predicted, expected);

            REQUIRE(observedOutput == Catch::Approx(expectedOutput));
    
            Matrix predicted2 = {{ 0.0 }, { 0.0 }};
            Matrix expected2 = {{ 0.0 }, { 0.0 }};
            observedOutput = LossFunctionImplementation::evaluateSumSquaredError(predicted2, expected2);

            REQUIRE(observedOutput == Catch::Approx(0.0));
        }
    
        SECTION("SSE DERIVATIVE") {
            Matrix predicted = {{ 3.0 }, { 1.0 }};
            Matrix expected = {{ 1.0 }, { 0.0 }};
            Matrix observedOutput = LossFunctionImplementation::evaluateSumSquaredErrorDerivative(predicted, expected);
    
            for (int r = 0;r<2;r++) {
                auto expectedValue = 2.0 * (predicted(r, 0) - expected(r, 0));

                REQUIRE(observedOutput(r, 0) == Catch::Approx(expectedValue));
            }
    
            Matrix predicted2 = {{ -1.0 }, { 2.0 }};
            Matrix expected2 = {{ -1.0 }, { 2.0 }};
            observedOutput = LossFunctionImplementation::evaluateSumSquaredErrorDerivative(predicted2, expected2);
    
            for (int r = 0;r<2;r++) REQUIRE(observedOutput(r, 0) == Catch::Approx(0.0));
        }
    
        SECTION("EVALUATE BCE") {
            Matrix predicted = {{ 0.9 }, { 0.1 }};
            Matrix expected = {{ 1.0 }, { 0.0 }};
            auto observedOutput = LossFunctionImplementation::evaluateBinaryCrossEntropy(predicted, expected);
    
            auto expectedOutput = - ( 1.0 * std::log(0.9) + (1.0 - 1.0) * std::log(1.0 - 0.9) )
                                   - ( 0.0 * std::log(0.1) + (1.0 - 0.0) * std::log(1.0 - 0.1) );

            REQUIRE(observedOutput == Catch::Approx(expectedOutput));
    
            Matrix predicted2 = {{ 0.999999 }, { 1e-8 }};
            Matrix expected2 = {{ 1.0 }, { 0.0 }};
            observedOutput = LossFunctionImplementation::evaluateBinaryCrossEntropy(predicted2, expected2);
    
            expectedOutput = - ( 1.0 * std::log(0.999999) + (1.0 - 1.0) * std::log(1.0 - 0.999999) )
                             - ( 0.0 * std::log(1e-7f) + (1.0 - 0.0) * std::log(1.0 - 1e-7f) );

            REQUIRE(observedOutput == Catch::Approx(expectedOutput).margin(1e-5));
        }
    
        SECTION("BCE DERIVATIVE") {
            Matrix predicted = {{ 0.8 }, { 0.2 }};
            Matrix expected = {{ 1.0 }, { 0.0 }};
            Matrix observedOutput = LossFunctionImplementation::evaluateBinaryCrossEntropyDerivative(predicted, expected);
    
            for (int r = 0;r<2;r++) {
                auto predVal = std::min(std::max(predicted(r, 0), 1e-7f), 1.0f - 1e-7f);
                auto expectedVal = expected(r, 0);
                auto expectedValue = (predVal - expectedVal) / (predVal * (1.0 - predVal));

                REQUIRE(observedOutput(r, 0) == Catch::Approx(expectedValue));
            }
    
            Matrix predicted2 = {{ 1e-8 }, { 0.999999 }};
            Matrix expected2 = {{ 0.0 }, { 1.0 }};
            observedOutput = LossFunctionImplementation::evaluateBinaryCrossEntropyDerivative(predicted2, expected2);
    
            for (int r = 0;r<2;r++) {
                auto predVal = std::min(std::max(predicted2(r, 0), 1e-7f), 1.0f - 1e-7f);
                auto expectedVal = expected2(r, 0);
                auto expectedValue = (predVal - expectedVal) / (predVal * (1.0 - predVal));

                REQUIRE(observedOutput(r, 0) == Catch::Approx(expectedValue));
            }
        }
    
        SECTION("EVALUATE CCE") {
            Matrix predicted = {{ 0.9 }, { 0.1 }};
            Matrix expected = {{ 1.0 }, { 0.0 }};
            auto observedOutput = LossFunctionImplementation::evaluateCategoricalCrossEntropy(predicted, expected);
    
            auto expectedOutput = - (1.0 * std::log(0.9)) - (0.0 * std::log(0.1));
            
            REQUIRE(observedOutput == Catch::Approx(expectedOutput));
    
            Matrix predicted2 = {{ 1.0 }, { 1e-7 }};
            Matrix expected2 = {{ 0.0 }, { 1.0 }};
            observedOutput = LossFunctionImplementation::evaluateCategoricalCrossEntropy(predicted2, expected2);
    
            expectedOutput = - (0.0 * std::log(1.0)) - (1.0 * std::log(1e-7f));

            REQUIRE(observedOutput == Catch::Approx(expectedOutput));
        }
    
        SECTION("CCE DERIVATIVE") {
            Matrix predicted = {{ 0.5 }, { 0.7 }};
            Matrix expected = {{ 1.0 }, { 0.0 }};
            Matrix observedOutput = LossFunctionImplementation::evaluateCategoricalCrossEntropyDerivative(predicted, expected);
    
            for (int r = 0;r<2;r++) {
                auto predVal = std::min(std::max(predicted(r, 0), 1e-7f), 1.0f);
                auto expectedVal = expected(r, 0);
                auto expectedValue = - expectedVal / predVal;

                REQUIRE(observedOutput(r, 0) == Catch::Approx(expectedValue));
            }
    
            Matrix predicted2 = {{ 1e-7 }, { 0.999999 }};
            Matrix expected2 = {{ 0.0 }, { 1.0 }};
            observedOutput = LossFunctionImplementation::evaluateCategoricalCrossEntropyDerivative(predicted2, expected2);
    
            for (int r = 0;r<2;r++) {
                auto predVal = std::min(std::max(predicted2(r, 0), 1e-7f), 1.0f);
                auto expectedVal = expected2(r, 0);
                auto expectedValue = - expectedVal / predVal;
                
                REQUIRE(observedOutput(r, 0) == Catch::Approx(expectedValue));
            }
        }
        
        SECTION("EVALUATE LOSS FUNCTION") {
            Matrix predicted = {{ 0.9f }, { 0.1f }};
            Matrix expected = {{ 1.0f }, { 0.0f }};
        
            auto mse = evaluateLoss(LossFunction::MEAN_SQUARED_ERROR, predicted, expected);
            auto expectedMse = ((0.1f * 0.1f) + (0.1f * 0.1f)) / 2.0f;

            REQUIRE(mse == Catch::Approx(expectedMse));
        
            auto sse = evaluateLoss(LossFunction::SUM_SQUARED_ERROR, predicted, expected);
            auto expectedSse = ((0.1f * 0.1f) + (0.1f * 0.1f));

            REQUIRE(sse == Catch::Approx(expectedSse));
        
            auto bce = evaluateLoss(LossFunction::BINARY_CROSS_ENTROPY, predicted, expected);
            auto expectedBce = 
                - (1.0f * std::log(0.9f) + (1.0f - 1.0f) * std::log(1.0f - 0.9f))
                - (0.0f * std::log(0.1f) + (1.0f - 0.0f) * std::log(1.0f - 0.1f));

            REQUIRE(bce == Catch::Approx(expectedBce));
        
            auto cce = evaluateLoss(LossFunction::CATEGORICAL_CROSS_ENTROPY, predicted, expected);
            auto expectedCce = - (1.0f * std::log(0.9f)) - (0.0f * std::log(0.1f));

            REQUIRE(cce == Catch::Approx(expectedCce));
        }

        SECTION("LOSS FUNCTION DERIVATIVE") {
            Matrix predicted = {{ 0.9f }, { 0.1f }};
            Matrix expected = {{ 1.0f }, { 0.0f }};

            Matrix mseDeriv = evaluateLossDerivative(LossFunction::MEAN_SQUARED_ERROR, predicted, expected);
            for (int r = 0;r<2;r++) {
                auto coeff = 2.0f / 2.0f;
                auto expectedVal = coeff * (predicted(r, 0) - expected(r, 0));

                REQUIRE(mseDeriv(r, 0) == Catch::Approx(expectedVal));
            }
        
            Matrix sseDeriv = evaluateLossDerivative(LossFunction::SUM_SQUARED_ERROR, predicted, expected);
            for (int r = 0;r<2;r++) {
                auto expectedVal = 2.0f * (predicted(r, 0) - expected(r, 0));

                REQUIRE(sseDeriv(r, 0) == Catch::Approx(expectedVal));
            }
        
            Matrix bceDeriv = evaluateLossDerivative(LossFunction::BINARY_CROSS_ENTROPY, predicted, expected);
            for (int r = 0;r<2;r++) {
                auto predVal = std::min(std::max(predicted(r, 0), 1e-7f), 1.0f - 1e-7f);
                auto expectedVal = expected(r, 0);
                auto expectedOutput = (predVal - expectedVal) / (predVal * (1.0f - predVal));

                REQUIRE(bceDeriv(r, 0) == Catch::Approx(expectedOutput));
            }
        
            Matrix cceDeriv = evaluateLossDerivative(LossFunction::CATEGORICAL_CROSS_ENTROPY, predicted, expected);
            for (int r = 0;r<2;r++) {
                auto predVal = std::min(std::max(predicted(r, 0), 1e-7f), 1.0f);
                auto expectedVal = expected(r, 0);
                auto expectedOutput = - expectedVal / predVal;

                REQUIRE(cceDeriv(r, 0) == Catch::Approx(expectedOutput));
            }
        }
    }

    SECTION("TRAINING CYCLE") {
        NeuralNetwork nn(
            2,
            {
                HiddenLayerParameters(RELU,
                    {
                        { 0.2, -0.4 },
                        { -0.1, 0.9 },
                        { 0.3, 0.6 }
                    },
                    {
                        { 0.0 },
                        { 0.7 },
                        { -0.3 }
                    }
                ),
                HiddenLayerParameters(SIGMOID,
                    {
                        { 0.8, -0.6, -0.5 },
                        { 0.2, 0.3, 0.1 }
                    },
                    {
                        { 0.05 },
                        { -0.1 }
                    }
                )
            },
            SOFTMAX,
            CATEGORICAL_CROSS_ENTROPY
        );

        DataPoint trainingDataPoint(
            {{ 0.5 }, { -0.3 }},
            {{ 0.01 }, { 0.99 }}
        );

        nn.train(trainingDataPoint, 1.0);

        auto observedHiddenLayerStates = nn.getHiddenLayerStates();
        auto observedHiddenLayerParameters = nn.getHiddenLayerParameters();

        SECTION("FEED FORWARD") {
            SECTION("HIDDEN LAYER 0") {
                Matrix expectedActivated({{ 0.22 }, { 0.38 }, { 0.0 }});
                auto observedActivated = observedHiddenLayerStates[0].activated;

                REQUIRE(matricesAreApproxEqual(expectedActivated, observedActivated));
            }

            SECTION("HIDDEN LAYER 1") {
                Matrix expectedActivated = {{ 0.4995 }, { 0.5145 }};
                auto observedActivated = observedHiddenLayerStates[1].activated;

                REQUIRE(matricesAreApproxEqual(expectedActivated, observedActivated));
            }

            SECTION("OUTPUT LAYER") {
                Matrix expectedNormalized = {{ 0.4963 }, { 0.5037 }};
                auto observedNormalized = nn.getNormalizedOutput();

                REQUIRE(matricesAreApproxEqual(expectedNormalized, observedNormalized));
            }
        }

        SECTION("BACK PROPAGATION") {
            SECTION("HIDDEN LAYER 1") {
                Matrix expectedActivatedPartial = {{ 0.4863 }, { -0.4863 }};
                auto observedActivatedPartial = observedHiddenLayerStates[1].dLossWrtActivated;

                REQUIRE(matricesAreApproxEqual(expectedActivatedPartial, observedActivatedPartial));

                Matrix expectedBiasedPartial = {{ 0.1216 }, { -0.1215 }};
                auto observedBiasedPartial = observedHiddenLayerStates[1].dLossWrtBiased;

                REQUIRE(matricesAreApproxEqual(expectedBiasedPartial, observedBiasedPartial));

                Matrix expectedWeightsPartial({{ 0.0267, 0.0462, 0.0 }, { -0.0267, -0.0462, 0.0 }});
                auto observedWeightsPartial = observedHiddenLayerStates[1].dLossWrtWeights;

                REQUIRE(matricesAreApproxEqual(expectedWeightsPartial, observedWeightsPartial));
            }

            SECTION("HIDDEN LAYER 0") {
                Matrix expectedActivatedPartial({{ 0.0729 }, { -0.1094 }, { -0.0729 }});
                auto observedActivatedPartial = observedHiddenLayerStates[0].dLossWrtActivated;

                REQUIRE(matricesAreApproxEqual(expectedActivatedPartial, observedActivatedPartial));

                Matrix expectedBiasedPartial({{ 0.0729 }, { -0.1094 }, { 0.0 }});
                auto observedBiasedPartial = observedHiddenLayerStates[0].dLossWrtBiased;

                REQUIRE(matricesAreApproxEqual(expectedBiasedPartial, observedBiasedPartial));

                Matrix expectedWeightsPartial({{ 0.0365, -0.0219 }, { -0.0547, 0.0328 }, { 0.0, 0.0 }});
                auto observedWeightsPartial = observedHiddenLayerStates[0].dLossWrtWeights;

                REQUIRE(matricesAreApproxEqual(expectedWeightsPartial, observedWeightsPartial));

                Matrix expectedInputPartial = {{ 0.0255 }, { -0.1276 }};
                auto observedInputPartial = observedHiddenLayerStates[0].dLossWrtInput;

                REQUIRE(matricesAreApproxEqual(expectedInputPartial, observedInputPartial));
            }
        }

        SECTION("ADJUSTMENTS") {
            SECTION("HIDDEN LAYER 0") {
                Matrix expectedWeights({{ 0.1635, -0.3781 }, { -0.0453, 0.8672 }, { 0.3, 0.6 }});
                auto observedWeights = observedHiddenLayerParameters[0].weights;

                REQUIRE(matricesAreApproxEqual(expectedWeights, observedWeights));

                Matrix expectedBias({{ -0.0730 }, { 0.8094 }, { -0.3 }});
                auto observedBias = observedHiddenLayerParameters[0].bias;

                REQUIRE(matricesAreApproxEqual(expectedBias, observedBias));
            }

            SECTION("HIDDEN LAYER 1") {
                Matrix expectedWeights({{ 0.7733, -0.6462, -0.5 }, { 0.2267, 0.3462, 0.1 }});
                auto observedWeights = observedHiddenLayerParameters[1].weights;

                REQUIRE(matricesAreApproxEqual(expectedWeights, observedWeights));

                Matrix expectedBias = {{ -0.0716 }, { 0.0215 }};
                auto observedBias = observedHiddenLayerParameters[1].bias;

                REQUIRE(matricesAreApproxEqual(expectedBias, observedBias));
            }
        }
    }
}