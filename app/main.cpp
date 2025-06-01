#include "./examples/curve_fitting.hpp"

int main()
{
    polynomialFitParametricCurve(
        "sin(x)",
        std::function<float(float)>([](float x) -> float { return sin(x); }),
        -2.0,
        2.0,
        200,
        7
    );

    polynomialFitParametricCurve(
        "cos(x)",
        std::function<float(float)>([](float x) -> float { return cos(x); }),
        -2.0,
        2.0,
        200,
        8
    );
};