#include "Layer.h"
#include <stdexcept>

namespace neural_autodiff {

    LinearLayer::LinearLayer(size_t in_features, size_t out_features)
        : in_features_(in_features)
        , out_features_(out_features) {

        Matrix weight_matrix(out_features, in_features);
        weight_matrix.xavier_init();
        weights_ = Node::make_parameter(weight_matrix);

        Matrix bias_matrix(out_features, 1);
        bias_matrix.zeros();
        bias_ = Node::make_parameter(bias_matrix);
    }

    NodePtr LinearLayer::forward(NodePtr input) {
        if (!input) {
            throw std::invalid_argument("Null input to linear layer");
        }

        if (input->value_.cols != 1) {
            throw std::invalid_argument("Input must be a column vector");
        }

        if (input->value_.rows != in_features_) {
            throw std::invalid_argument("Input features dimension mismatch");
        }

        // Y = WX + b
        NodePtr output = Node::matmul(weights_, input);
        output = Node::add(output, bias_);

        return output;
    }

    std::vector<NodePtr> LinearLayer::parameters() const {
        return {weights_, bias_};
    }

    void LinearLayer::zero_grad() {
        weights_->zero_grad();
        bias_->zero_grad();
    }

}