#include "Layer.h"

#include <iostream>
#include <stdexcept>

namespace neural_autodiff {

    LinearLayer::LinearLayer(int in_features, int out_features)
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
        std::cout << "In layer " << output->value_.rows << " "<<output->value_.rows <<std::endl;
        return output;
    }

    void backward(NodePtr output, const Matrix& gradient) {

        if (!output) {
            throw std::invalid_argument("Null node in backward pass");
        }

        if (gradient.rows != output->value_.rows || gradient.cols != output->value_.cols) {
            throw std::invalid_argument("Gradient dimensions don't match node dimensions");
        }

        for (int i = 0; i < output->grad_.rows; ++i) {
            for (int j = 0; j < output->grad_.cols; ++j) {
                output->grad_.at(i, j) += gradient.at(i, j);
            }
        }

        if (output->op_type() == OpType::ADD) {
            if (output->inputs_.size() != 2) {
                throw std::runtime_error("ADD type Nodeptr must have 2 inputs");
            }

            auto a = output->inputs_[0];
            auto b = output->inputs_[1];

            backward(a, gradient);
            backward(b, gradient);
        }

         else if (output->op_type() == OpType::MATMUL) {
             if (output->inputs_.size() != 2) {
                 throw std::runtime_error("Matmul type nodeptr must have 2 inputs");
             }

             auto a = output->inputs_[0];
             auto b = output->inputs_[1];

             Matrix grad_a  = Matrix::multiply(gradient, Matrix::transpose(b->value_));
             backward(a, grad_a);

             Matrix grad_b = Matrix::multiply(Matrix::transpose(a->value_), gradient);
             backward(b, grad_b);
         }

         else if (output->op_type() == OpType::RELU) {
             if (output->inputs_.size() != 1) {
                 throw std::runtime_error("Relu type nodeptr must have 1 inputs");
             }

             Matrix grad_relu(gradient.rows, gradient.cols);

             grad_relu.zeros();

             auto input = output->inputs_[0];

             for (int i = 0; i < gradient.rows; ++i) {
                 for (int j = 0; j < gradient.cols; ++j) {

                     grad_relu.at(i, j) = gradient.at(i, j);

                     if (gradient.at(i, j) > 0) {
                         grad_relu.at(i, j) = gradient.at(i, j);
                     }
                 }
             }

             backward(input , grad_relu);
         }

         else if (output->op_type() == OpType::SIGMOID) {
             if (output->inputs_.size() != 1) {
                 throw std::runtime_error("Sigmoid type nodeptr must have 1 inputs");
             }

             auto input = output->inputs_[0];

             Matrix grad_sigmoid(gradient.rows, gradient.cols);
             grad_sigmoid.zeros();

             for (int i = 0; i < gradient.rows; ++i) {
                 for (int j = 0; j < gradient.cols; ++j) {
                     double y  = output->inputs_[0]->value_.at(i, j);
                     grad_sigmoid.at(i, j) = gradient.at(i, j) * (y * (1 - y));
                 }
             }

             backward(input, grad_sigmoid);
         }

         else if (output->op_type() == OpType::TANH) {
             if (output->inputs().size() != 1) {
                 throw std::runtime_error("Sigmoid node must have exactly 1 input");
             }

             Matrix grad_sigmoid(gradient.rows, gradient.cols);
             auto input = output->inputs()[0];

             for (int i = 0; i < gradient.rows; ++i) {
                 for (int j = 0; j < gradient.cols; ++j) {
                     double y = output->value_.at(i, j);
                     grad_sigmoid.at(i, j) = gradient.at(i, j) * (1 - y * y);
                 }
             }
             backward(input, grad_sigmoid);
         }

         else {
             return;
         }
    }


    void LinearLayer::zero_grad() {
        weights_->zero_grad();
        bias_->zero_grad();
    }

}