
#ifndef LAYER_H
#define LAYER_H

#include "Node.h"

namespace neural_autodiff {

    class LinearLayer{
    public:
        NodePtr weights_;
        NodePtr bias_;

        int in_features_;
        int out_features_;

        LinearLayer(int in_features, int out_features);
        LinearLayer(int in_features, int out_features, const Matrix& weights, const Matrix& biases);
        NodePtr forward(NodePtr input);
        void backward(NodePtr output_gradient, double learning_rate);

        void zero_grad();

    };

}
#endif //LAYER_H