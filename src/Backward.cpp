#include "Backward.h"
#include <queue>
#include <stdexcept>

namespace neural_autodiff {

void backward(NodePtr node, const Matrix& gradient) {
    if (!node) {
        throw std::invalid_argument("Null node in backward pass");
    }

    if (gradient.rows != node->value_.rows || gradient.cols != node->value_.cols) {
        throw std::invalid_argument("Gradient dimensions don't match node dimensions");
    }

    for (int i = 0; i < node->grad_.rows; ++i) {
        for (int j = 0; j < node->grad_.cols; ++j) {
            node->grad_.at(i, j) += gradient.at(i, j);
        }
    }

    switch (node->op_type()) {
        case OpType::MATMUL: {
            if (node->inputs().size() != 2) {
                throw std::runtime_error("MatMul node must have exactly 2 inputs");
            }

            auto a = node->inputs()[0];
            auto b = node->inputs()[1];

            Matrix grad_a = Matrix::multiply(gradient, Matrix::transpose(b->value_));
            backward(a, grad_a);

            Matrix grad_b = Matrix::multiply(Matrix::transpose(a->value_), gradient);
            backward(b, grad_b);
            break;
        }

        case OpType::ADD: {
            if (node->inputs().size() != 2) {
                throw std::runtime_error("Add node must have exactly 2 inputs");
            }

            backward(node->inputs()[1], gradient);
            backward(node->inputs()[0], gradient);
            break;
        }

        case OpType::RELU: {
            if (node->inputs().size() != 1) {
                throw std::runtime_error("ReLU node must have exactly 1 input");
            }

            Matrix relu_grad(gradient.rows, gradient.cols);
            auto input = node->inputs()[0];

            for (int i = 0; i < input->value_.rows; ++i) {
                for (int j = 0; j < input->value_.cols; ++j) {
                    relu_grad.at(i, j) = input->value_.at(i, j) > 0 ? gradient.at(i, j) : 0;
                }
            }

            backward(input, relu_grad);
            break;
        }

        case OpType::SIGMOID: {
            if (node->inputs().size() != 1) {
                throw std::runtime_error("Sigmoid node must have exactly 1 input");
            }

            Matrix sigmoid_grad(gradient.rows, gradient.cols);
            auto input = node->inputs()[0];

            for (int i = 0; i < node->value_.rows; ++i) {
                for (int j = 0; j < node->value_.cols; ++j) {
                    double y = node->value_.at(i, j);
                    sigmoid_grad.at(i, j) = gradient.at(i, j) * y * (1 - y);
                }
            }

            backward(input, sigmoid_grad);
            break;
        }

        case OpType::TANH: {
            if (node->inputs().size() != 1) {
                throw std::runtime_error("Tanh node must have exactly 1 input");
            }

            Matrix tanh_grad(gradient.rows, gradient.cols);
            auto input = node->inputs()[0];

            for (int i = 0; i < node->value_.rows; ++i) {
                for (int j = 0; j < node->value_.cols; ++j) {
                    double y = node->value_.at(i, j);
                    tanh_grad.at(i, j) = gradient.at(i, j) * (1 - y * y);
                }
            }

            backward(input, tanh_grad);
            break;
        }

        case OpType::INPUT:
        case OpType::PARAMETER:
            break;

        default:
            throw std::runtime_error("Invalid operation. Known operations are: ADD, SUBTRACT, MULTIPLICATION, TANH, RELU, SIGMOID");
    }
}

}