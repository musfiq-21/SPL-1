#include "Activation.h"
#include <cmath>
#include <stdexcept>

namespace neural_autodiff {

    NodePtr Activation::relu(NodePtr x) {
        if (!x) {
            throw std::invalid_argument("Null input to ReLU activation");
        }

        Matrix result(x->value_.rows, x->value_.cols);

        for (int i = 0; i < x->value_.rows; ++i) {
            for (int j = 0; j < x->value_.cols; ++j) {
                result.at(i, j) = std::max(0.0, x->value_.at(i, j));
            }
        }

        auto node = std::make_shared<Node>(result, OpType::RELU);
        node->inputs_.push_back(x);

        return node;
    }

    NodePtr Activation::sigmoid(NodePtr x) {
        if (!x) {
            throw std::invalid_argument("Null input to sigmoid activation");
        }

        Matrix result(x->value_.rows, x->value_.cols);

        for (int i = 0; i < x->value_.rows; ++i) {
            for (int j = 0; j < x->value_.cols; ++j) {
                double val = x->value_.at(i, j);
                val = std::max(-500.0, std::min(500.0, val));
                result.at(i, j) = 1.0 / (1.0 + std::exp(-val));
            }
        }

        auto node = std::make_shared<Node>(result, OpType::SIGMOID);
        node->inputs_.push_back(x);

        return node;
    }

    NodePtr Activation::tanh(NodePtr x) {
        if (!x) {
            throw std::invalid_argument("Null input to tanh activation");
        }

        Matrix result(x->value_.rows, x->value_.cols);

        for (size_t i = 0; i < x->value_.rows; ++i) {
            for (size_t j = 0; j < x->value_.cols; ++j) {
                double val = x->value_.at(i, j);

                result.at(i, j) = std::tanh(val);
            }
        }

        auto node = std::make_shared<Node>(result, OpType::TANH);
        node->inputs_.push_back(x);

        return node;
    }

}