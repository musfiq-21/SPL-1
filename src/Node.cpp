#include "Node.h"
#include <stdexcept>

namespace neural_autodiff {

    Node::Node(const Matrix& value, OpType op)
        : value_(value)
        , grad_(value.rows, value.cols)
        , op_type_(op) {
        grad_.zeros();
    }

    void Node::zero_grad() {
        grad_.zeros();
    }

    std::shared_ptr<Node> Node::make_parameter(const Matrix& value) {
        return std::make_shared<Node>(value, OpType::PARAMETER);
    }

    std::shared_ptr<Node> Node::make_input(const Matrix& value) {
        return std::make_shared<Node>(value, OpType::INPUT);
    }

    std::shared_ptr<Node> Node::matmul(std::shared_ptr<Node> a, std::shared_ptr<Node> b) {
        if (!a or !b) {
            throw std::invalid_argument("Null node pointer in matrix multiplication operation");
        }

        Matrix result = Matrix::multiply(a->value_, b->value_);
        auto node = std::make_shared<Node>(result, OpType::MATMUL);

        node->inputs_.push_back(a);
        node->inputs_.push_back(b);

        return node;
    }

    std::shared_ptr<Node> Node::add(std::shared_ptr<Node> a, std::shared_ptr<Node> b) {
        if (!a or !b) {
            throw std::invalid_argument("Null node pointer in add operation");
        }

        if (a->value_.rows != b->value_.rows || a->value_.cols != b->value_.cols) {
            throw std::invalid_argument("Incompatible dimensions for addition");
        }

        Matrix result = Matrix::add(a->value_, b->value_);
        auto node = std::make_shared<Node>(result, OpType::ADD);

        node->inputs_.push_back(a);
        node->inputs_.push_back(b);

        return node;
    }

    void Node::backward(const Matrix& grad_output) {
        // Add incoming gradient to node's gradient
        if (grad_.rows != grad_output.rows || grad_.cols != grad_output.cols) {
            throw std::invalid_argument("Incompatible gradient dimensions");
        }
        grad_ = Matrix::add(grad_, grad_output);

        // Base case: no inputs to propagate to
        if (inputs_.empty()) {
            return;
        }

        // Propagate gradients based on operation type
        switch (op_type_) {
            case OpType::MATMUL: {
                if (inputs_.size() != 2) {
                    throw std::runtime_error("MATMUL node must have exactly 2 inputs");
                }

                auto a = inputs_[0];
                auto b = inputs_[1];

                // Gradient w.r.t first input (a): grad_output * b^T
                Matrix b_T = b->value_.transpose();
                Matrix grad_a = Matrix::multiply(grad_output, b_T);
                a->backward(grad_a);

                // Gradient w.r.t second input (b): a^T * grad_output
                Matrix a_T = a->value_.transpose();
                Matrix grad_b = Matrix::multiply(a_T, grad_output);
                b->backward(grad_b);
                break;
            }

            case OpType::ADD: {
                if (inputs_.size() != 2) {
                    throw std::runtime_error("ADD node must have exactly 2 inputs");
                }

                // Addition passes gradient unchanged to both inputs
                for (auto& input : inputs_) {
                    input->backward(grad_output);
                }
                break;
            }

            case OpType::PARAMETER:
            case OpType::INPUT:
                // Leaf nodes - gradient accumulation stops here
                break;

            default:
                throw std::runtime_error("Backward not implemented for this operation");
        }
    }



}