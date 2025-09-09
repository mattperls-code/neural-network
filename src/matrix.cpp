#include "matrix.hpp"

#include <xtensor/generators/xrandom.hpp>

#include <stdexcept>
#include <format>

Matrix randomMatrix(int rows, int cols, float min, float max)
{
    return xt::random::rand<float>({ rows, cols }) * (max - min) + min;
};

std::string matrixToStr(const Matrix& mat)
{
    std::ostringstream oss;

    oss << "{\n";

    for (size_t i = 0; i < mat.shape()[0]; ++i) {
        oss << "  {";

        for (size_t j = 0; j < mat.shape()[1]; ++j) {
            oss << mat(i, j);

            if (j + 1 < mat.shape()[1]) oss << ", ";
        }

        oss << "}";

        if (i + 1 < mat.shape()[0]) oss << ",";

        oss << "\n";
    }
    
    oss << "}";

    return oss.str();
};