#include "LossFunction.h"
#include <cmath>
#include <stdexcept>
#include <numeric>

namespace neural_autodiff {

    NodePtr Loss::mse_loss(NodePtr predicted, NodePtr target) {
        if (!predicted || !target) {
            throw std::invalid_argument("Null input to MSE loss");
        }

        if (predicted->value_.rows != target->value_.rows
            or
            predicted->value_.cols != target->value_.cols
            )

        {

            throw std::invalid_argument("Dimension mismatch in MSE loss");
        }

        Matrix diff_matrix(predicted->value_.rows, predicted->value_.cols);
        double sum_squared_error = 0.0;
        int n = predicted->value_.rows * predicted->value_.cols;

        for (int i = 0; i < predicted->value_.rows; ++i) {
            for (int j = 0; j < predicted->value_.cols; ++j) {
                double diff = predicted->value_.at(i, j) - target->value_.at(i, j);
                diff_matrix.at(i, j) = diff * diff;
                sum_squared_error += diff * diff;
            }
        }

        Matrix result(1, 1);
        result.at(0, 0) = sum_squared_error / static_cast<double>(n);

        auto node = std::make_shared<Node>(result, OpType::INPUT);
        node->inputs_.push_back(predicted);
        node->inputs_.push_back(target);
        return node;
    }

    NodePtr Loss::mse_loss_prime(NodePtr predicted, NodePtr target) {
        if (!predicted || !target) {
            throw std::invalid_argument("Null pointer in MSE loss prime");
        }

        if (predicted->value_.rows != target->value_.rows
            or
            predicted->value_.cols != target->value_.cols
            )

        {

            throw std::invalid_argument("Dimension mismatch in MSE loss prime");
        }
    }

}