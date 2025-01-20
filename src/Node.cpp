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

        // Store input nodes for backward pass
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

        // Store input nodes for backward pass
        node->inputs_.push_back(a);
        node->inputs_.push_back(b);

        return node;
    }

}