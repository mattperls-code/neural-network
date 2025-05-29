#include <catch2/catch_all.hpp>

#include "../src/matrix.hpp"

TEST_CASE("MATRIX") {

    SECTION("transpose") {
        Matrix input1({{1, 2}, {3, 4}});
        Matrix expectedOutput1({{1, 3}, {2, 4}});
        Matrix observedOutput1 = Matrix::transpose(input1);

        REQUIRE(observedOutput1.toString() == expectedOutput1.toString());

        Matrix input2({{5, 6, 7}});
        Matrix expectedOutput2({{5}, {6}, {7}});
        Matrix observedOutput2 = Matrix::transpose(input2);

        REQUIRE(observedOutput2.toString() == expectedOutput2.toString());
    }

    SECTION("add") {
        Matrix inputA1({{1, 2}, {3, 4}});
        Matrix inputB1({{5, 6}, {7, 8}});
        Matrix expectedOutput1({{6, 8}, {10, 12}});
        Matrix observedOutput1 = Matrix::add(inputA1, inputB1);

        REQUIRE(observedOutput1.toString() == expectedOutput1.toString());

        Matrix inputA2({{0}});
        Matrix inputB2({{0}});
        Matrix expectedOutput2({{0}});
        Matrix observedOutput2 = Matrix::add(inputA2, inputB2);

        REQUIRE(observedOutput2.toString() == expectedOutput2.toString());
    }

    SECTION("subtract") {
        Matrix inputA1({{5, 6}, {7, 8}});
        Matrix inputB1({{1, 2}, {3, 4}});
        Matrix expectedOutput1({{4, 4}, {4, 4}});
        Matrix observedOutput1 = Matrix::subtract(inputA1, inputB1);

        REQUIRE(observedOutput1.toString() == expectedOutput1.toString());

        Matrix inputA2({{10}});
        Matrix inputB2({{5}});
        Matrix expectedOutput2({{5}});
        Matrix observedOutput2 = Matrix::subtract(inputA2, inputB2);

        REQUIRE(observedOutput2.toString() == expectedOutput2.toString());
    }

    SECTION("scalarProduct") {
        Matrix inputMat1({{1, -2}, {0, 4}});
        float inputScalar1 = 2.0;
        Matrix expectedOutput1({{2, -4}, {0, 8}});
        Matrix observedOutput1 = Matrix::scalarProduct(inputMat1, inputScalar1);

        REQUIRE(observedOutput1.toString() == expectedOutput1.toString());

        Matrix inputMat2({{3, 3}, {3, 3}});
        float inputScalar2 = 0.0;
        Matrix expectedOutput2({{0, 0}, {0, 0}});
        Matrix observedOutput2 = Matrix::scalarProduct(inputMat2, inputScalar2);

        REQUIRE(observedOutput2.toString() == expectedOutput2.toString());
    }

    SECTION("hadamardProduct") {
        Matrix inputA1({{1, 2}, {3, 4}});
        Matrix inputB1({{2, 0}, {1, 2}});
        Matrix expectedOutput1({{2, 0}, {3, 8}});
        Matrix observedOutput1 = Matrix::hadamardProduct(inputA1, inputB1);

        REQUIRE(observedOutput1.toString() == expectedOutput1.toString());

        Matrix inputA2({{0, 1}, {1, 0}});
        Matrix inputB2({{1, 0}, {0, 1}});
        Matrix expectedOutput2({{0, 0}, {0, 0}});
        Matrix observedOutput2 = Matrix::hadamardProduct(inputA2, inputB2);

        REQUIRE(observedOutput2.toString() == expectedOutput2.toString());
    }

    SECTION("matrixProduct") {
        Matrix inputA1({{1, 2}, {3, 4}});
        Matrix inputB1({{2, 0}, {1, 2}});
        Matrix expectedOutput1({{4, 4}, {10, 8}});
        Matrix observedOutput1 = Matrix::matrixProduct(inputA1, inputB1);

        REQUIRE(observedOutput1.toString() == expectedOutput1.toString());

        Matrix inputA2({{1, 0}, {0, 1}});
        Matrix inputB2({{5, 6}, {7, 8}});
        Matrix expectedOutput2({{5, 6}, {7, 8}});
        Matrix observedOutput2 = Matrix::matrixProduct(inputA2, inputB2);

        REQUIRE(observedOutput2.toString() == expectedOutput2.toString());
    }

    SECTION("matrixColumnProduct") {
        Matrix inputA1({{1, 2}, {3, 4}, {5, 6}});
        Matrix inputCol1(std::vector<std::vector<float>>({{1}, {2}}));
        Matrix expectedOutput1({{5}, {11}, {17}});
        Matrix observedOutput1 = Matrix::matrixColumnProduct(inputA1, inputCol1);

        REQUIRE(observedOutput1.toString() == expectedOutput1.toString());

        Matrix inputA2({{0, 1}, {1, 0}});
        Matrix inputCol2(std::vector<std::vector<float>>({{1}, {1}}));
        Matrix expectedOutput2(std::vector<std::vector<float>>({{1}, {1}}));
        Matrix observedOutput2 = Matrix::matrixColumnProduct(inputA2, inputCol2);

        REQUIRE(observedOutput2.toString() == expectedOutput2.toString());
    }
}