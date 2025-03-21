
#ifndef NODE_H
#define NODE_H
#include <memory>
#include <vector>
#include "Matrix.h"

namespace neural_autodiff {

    enum class OpType {
        INPUT,  PARAMETER,  ADD,    MULTIPLY,   MATMUL,
        TRANSPOSE,  RELU,   SIGMOID,    TANH
    };

    class Node {
    public:
        Matrix value_;
        Matrix grad_;
        OpType op_type_;
        std::vector<std::shared_ptr<Node>> inputs_;
        Node(const Matrix& value, OpType op = OpType::PARAMETER);

        void zero_grad();

        static std::shared_ptr<Node> make_parameter(const Matrix& value);
        static std::shared_ptr<Node> make_input(const Matrix& value);

        static std::shared_ptr<Node> matmul(std::shared_ptr<Node> a, std::shared_ptr<Node> b);
        static std::shared_ptr<Node> add(std::shared_ptr<Node> a, std::shared_ptr<Node> b);

        OpType op_type() const { return op_type_; }
        const std::vector<std::shared_ptr<Node>>& inputs() const { return inputs_; }

    };

    using NodePtr = std::shared_ptr<Node>;

}
#endif //NODE_H