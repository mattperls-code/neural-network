#include "matrix.hpp"

#include <stdexcept>
#include <format>

Matrix::Matrix(const Shape& shape)
{
    this->rows = shape.rows;
    this->cols = shape.cols;

    this->data = std::vector<float>(this->rows * this->cols, 0.0);
};

Matrix::Matrix(const Shape& shape, float defaultValue)
{
    this->rows = shape.rows;
    this->cols = shape.cols;

    this->data = std::vector<float>(this->rows * this->cols, defaultValue);
};

Matrix::Matrix(const Shape& shape, std::mt19937& rng, std::uniform_real_distribution<float>& defaultValueDistribution)
{
    this->rows = shape.rows;
    this->cols = shape.cols;

    int numElements = this->rows * this->cols;

    this->data = std::vector<float>(numElements);

    for (int i = 0;i<numElements;i++) this->data[i] = defaultValueDistribution(rng);
};

Matrix::Matrix(const std::vector<std::vector<float>>& mat)
{
    this->rows = mat.size();
    this->cols = mat.size() == 0 ? 0 : mat[0].size();

    this->data = std::vector<float>(this->rows * this->cols, 0.0);

    for (int r = 0;r<this->rows;r++) for (int c = 0;c<this->cols;c++) this->data[r * this->cols + c] = mat[r][c];
};

int Matrix::rowCount() const
{
    return this->rows;
};

int Matrix::colCount() const
{
    return this->cols;
};

bool Matrix::empty() const
{
    return this->rows == 0 || this->cols == 0;
};

Shape Matrix::shape() const
{
    return Shape(this->rows, this->cols);
};

float Matrix::get(int row, int col) const
{
    if (row < 0 || row > this->rows) throw std::runtime_error("Matrix get: row out of bounds");
    if (col < 0 || col > this->cols) throw std::runtime_error("Matrix get: col out of bounds");

    return this->data[row * this->cols + col];
};

void Matrix::set(int row, int col, float value)
{
    if (row < 0 || row > this->rows) throw std::runtime_error("Matrix set: row out of bounds");
    if (col < 0 || col > this->cols) throw std::runtime_error("Matrix set: col out of bounds");

    this->data[row * this->cols + col] = value;
};

int& Matrix::dangerouslyGetRows()
{
    return this->rows;
};

int& Matrix::dangerouslyGetCols()
{
    return this->cols;
};

std::vector<float>& Matrix::dangerouslyGetData()
{
    return this->data;
};

std::string Matrix::toString() const
{
    if (this->empty()) return "{ }";

    std::string output;

    output += "{ ";

    for (int r = 0;r<this->rows;r++) {
        output += "{ ";

        for (int c = 0;c<this->cols;c++) output += std::format("{:.5f}", this->get(r, c)) + ", ";

        output.pop_back();
        output.pop_back();

        output += " }, ";
    }

    output.pop_back();
    output.pop_back();

    output += " }";

    return output;
};

Matrix Matrix::transpose(const Matrix& mat) {
    Matrix output(Shape(mat.cols, mat.rows));

    for (int r = 0;r<mat.rows;r++) for (int c = 0;c<mat.cols;c++) output.set(c, r, mat.get(r, c));

    return output;
}

Matrix Matrix::add(const Matrix& matA, const Matrix& matB) {
    if (matA.rows != matB.rows) throw std::runtime_error("Matrix add: rowA != rowB");
    if (matA.cols != matB.cols) throw std::runtime_error("Matrix add: colA != colB");

    Matrix output(Shape(matA.rows, matA.cols));

    for (int i = 0;i<matA.data.size();i++) output.data[i] = matA.data[i] + matB.data[i];

    return output;
}

Matrix Matrix::subtract(const Matrix& matA, const Matrix& matB) {
    if (matA.rows != matB.rows) throw std::runtime_error("Matrix subtract: rowA != rowB");
    if (matA.cols != matB.cols) throw std::runtime_error("Matrix subtract: colA != colB");

    Matrix output(Shape(matA.rows, matA.cols));

    for (int i = 0;i<matA.data.size();i++) output.data[i] = matA.data[i] - matB.data[i];

    return output;
}

Matrix Matrix::scalarProduct(const Matrix& mat, float scalar) {
    Matrix output(Shape(mat.rows, mat.cols));

    for (int i = 0;i<mat.data.size();i++) output.data[i] = mat.data[i] * scalar;

    return output;
}

Matrix Matrix::hadamardProduct(const Matrix& matA, const Matrix& matB) {
    if (matA.rows != matB.rows) throw std::runtime_error("Matrix hadarmardProduct: rowA != rowB");
    if (matA.cols != matB.cols) throw std::runtime_error("Matrix hadarmardProduct: colA != colB");

    Matrix output(Shape(matA.rows, matA.cols));

    for (int i = 0;i<matA.data.size();i++) output.data[i] = matA.data[i] * matB.data[i];

    return output;
}

Matrix Matrix::matrixProduct(const Matrix& matA, const Matrix& matB) {
    if (matA.cols != matB.rows) throw std::runtime_error("Matrix matrixProduct: colA != rowB");

    Matrix output(Shape(matA.rows, matB.cols));

    for (int r = 0;r<matA.rows;r++) {
        for (int c = 0;c<matB.cols;c++) {
            float sum = 0.0;

            for (int i = 0;i<matA.cols;i++) sum += matA.get(r, i) * matB.get(i, c);

            output.set(r, c, sum);
        }
    }

    return output;
}

Matrix Matrix::matrixColumnProduct(const Matrix& mat, const Matrix& col) {
    if (col.cols != 1) throw std::runtime_error("Matrix matrixColumnProduct: col is not a column vector");
    if (mat.cols != col.rows) throw std::runtime_error("Matrix matrixColumnProduct: colMat != rowCol");

    Matrix output(Shape(mat.rows, 1));

    for (int r = 0;r<mat.rows;r++) {
        float sum = 0.0;

        for (int c = 0;c<mat.cols;c++) sum += mat.get(r, c) * col.get(c, 0);

        output.set(r, 0, sum);
    }
    
    return output;
}