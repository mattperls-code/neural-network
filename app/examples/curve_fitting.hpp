#ifndef EXAMPLES_CURVE_FITTING_HPP
#define EXAMPLES_CURVE_FITTING_HPP

#include <iostream>
#include <functional>

#include "../../src/neural_network.hpp"

void polynomialFit1DCurve(std::string curveName, std::function<float(float)> curveParametric, float minInput, float maxInput, int numDataPoints, int polynomialDegree);

#endif