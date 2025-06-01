#include "./examples/curve_fitting.hpp"

int main()
{
    polynomialFit1DCurve(
        "sin(x)",
        std::function<float(float)>([](float x) -> float { return sin(x); }),
        -1.0,
        1.0,
        100,
        7
    );
};