#include "Matrix.h"
#include <random>
#include <stdexcept>
#include <cmath>

namespace neural_autodiff {

Matrix::Matrix(int rows, int cols)
    : rows(rows), cols(cols), data(rows * cols, 0.0) {
}

Matrix::Matrix(int rows, int cols, const std::vector<double>& data)
    : rows(rows), cols(cols), data(data) {
    if (data.size() != rows * cols) {
        throw std::invalid_argument("Data size does not match matrix dimensions");
    }
}

Matrix Matrix::multiply(const Matrix& a, const Matrix& b) {
    if (a.cols != b.rows) {
        throw std::invalid_argument("Matrix dimensions incompatible for multiplication");
    }

    Matrix result(a.rows, b.cols);

    for (int i = 0; i < a.rows; ++i) {
        for (int j = 0; j < b.cols; ++j) {
            double sum = 0.0;
            for (int k = 0; k < a.cols; ++k) {
                sum += a.at(i, k) * b.at(k, j);
            }
            result.at(i, j) = sum;
        }
    }

    return result;
}

Matrix Matrix::transpose(const Matrix& a) {
    Matrix result(a.cols, a.rows);

    for (int i = 0; i < a.rows; ++i) {
        for (int j = 0; j < a.cols; ++j) {
            result.at(j, i) = a.at(i, j);
        }
    }

    return result;
}

Matrix Matrix::add(const Matrix& a, const Matrix& b) {
    if (a.rows != b.rows || a.cols != b.cols) {
        throw std::invalid_argument("Matrix dimensions must match for addition");
    }

    Matrix result(a.rows, a.cols);

    for (int i = 0; i < a.rows; ++i) {
        for (int j = 0; j < a.cols; ++j) {
            result.at(i, j) = a.at(i, j) + b.at(i, j);
        }
    }

    return result;
}

Matrix Matrix::subtract(const Matrix& a, const Matrix& b) {
    if (a.rows != b.rows || a.cols != b.cols) {
        throw std::invalid_argument("Matrix dimensions must match for subtraction");
    }

    Matrix result(a.rows, a.cols);

    for (int i = 0; i < a.rows; ++i) {
        for (int j = 0; j < a.cols; ++j) {
            result.at(i, j) = a.at(i, j) - b.at(i, j);
        }
    }

    return result;
}

void Matrix::xavier_init() {
    std::random_device rd;
    std::mt19937 gen(rd());

    // Xavier initialization scale
    double scale = std::sqrt(6.0 / (rows + cols));
    std::uniform_real_distribution<double> dis(-scale, scale);

    for (int i = 0; i < data.size(); ++i) {
        data[i] = dis(gen);
    }
}

void Matrix::zeros() {
    std::fill(data.begin(), data.end(), 0.0);
}

void Matrix::ones() {
    std::fill(data.begin(), data.end(), 1.0);
}

}