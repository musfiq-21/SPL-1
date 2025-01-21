#include "src/Node.h"
#include "src/Layer.h"
#include "src/Activation.h"
#include "src/LossFunction.h"
#include "src/Gradient_Descent.h"
#include "src/Backward.h"
#include <iostream>
#include <vector>

using namespace neural_autodiff;

double compute_accuracy(const std::vector<Matrix>& predictions, const std::vector<Matrix>& targets) {
    int correct = 0;
    for (int i = 0; i < predictions.size(); ++i) {
        bool pred = predictions[i].at(0, 0) >= 0.5;
        bool target = targets[i].at(0, 0) >= 0.5;
        if (pred == target) correct++;
    }
    return static_cast<double>(correct) / predictions.size();
}

int main() {

    std::vector<double> data_A  = {1, 2, -3, -4, 5, 6, 7, 8, 9, 10, 11, 12};
    std::vector<double> data_B  = {1, 1, 1, 1};

    Matrix a(3, 4, data_A);
    Matrix b(4, 1, data_B);

    auto node_a = Node::make_input(a);
    auto node_b = Node::make_input(b);

    auto node_c = Node::matmul(node_a, node_b);

    auto node_d = Activation::sigmoid(node_c);

    for (int i=0; i<node_d->value_.rows; ++i) {
        for (int j=0; j<node_d->value_.cols; ++j) {
            std::cout << node_d->value_.at(i,j) << " ";
        }
        std::cout << std::endl;
    }
    return 0;
}