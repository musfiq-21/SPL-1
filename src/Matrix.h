#pragma once
#include <vector>

namespace neural_autodiff {

    class Matrix {
    public:
        int rows;
        int cols;
        std::vector<double> data;
        
        Matrix(int rows, int cols);
        Matrix(int rows, int cols, const std::vector<double>& data);

        double& at(int i, int j) { return data[i * cols + j]; }
        const double& at(int i, int j) const { return data[i * cols + j]; }

        // Matrix operations
        static Matrix multiply(const Matrix& a, const Matrix& b);
        static Matrix transpose(const Matrix& a);
        static Matrix add(const Matrix& a, const Matrix& b);
        static Matrix subtract(const Matrix& a, const Matrix& b);


        void xavier_init();
        void zeros();
        void ones();

    };

}