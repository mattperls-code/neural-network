#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <vector>
#include <string>
#include <random>

class Shape
{
    public:
        int rows;
        int cols;

        Shape(int rows, int cols): rows(rows), cols(cols) {};

        bool operator==(const Shape&) const = default;
};

class Matrix
{
    private:
        int rows;
        int cols;
        std::vector<float> data;

    public:
        Matrix() = default;
        Matrix(const Shape& shape);
        Matrix(const Shape& shape, float defaultValue);
        Matrix(const Shape& shape, std::mt19937& rng, std::uniform_real_distribution<float>& defaultValueDistribution);
        Matrix(const std::vector<std::vector<float>>& mat);

        int rowCount() const;
        int colCount() const;
        bool empty() const;

        Shape shape() const;

        float get(int row, int col) const;
        void set(int row, int col, float value);

        int& dangerouslyGetRows();
        int& dangerouslyGetCols();
        std::vector<float>& dangerouslyGetData();

        std::string toString() const;

        static Matrix transpose(const Matrix& mat);
        static Matrix add(const Matrix& matA, const Matrix& matB);
        static Matrix subtract(const Matrix& matA, const Matrix& matB);
        static Matrix scalarProduct(const Matrix& mat, float scalar);
        static Matrix hadamardProduct(const Matrix& matA, const Matrix& matB);
        static Matrix matrixProduct(const Matrix& matA, const Matrix& matB);
        static Matrix matrixColumnProduct(const Matrix& mat, const Matrix& col);
};

#endif