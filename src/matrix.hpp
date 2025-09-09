#ifndef SRC_MATRIX
#define SRC_MATRIX

#include <string>

#include <xtensor/containers/xtensor.hpp>
#include <xtensor/utils/xtensor_simd.hpp>

using Matrix = xt::xtensor<float, 2>;

Matrix randomMatrix(int rows, int cols, float min, float max);

std::string matrixToStr(const Matrix& mat);

#endif