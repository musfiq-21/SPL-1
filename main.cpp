#include "src/Layer.h"
#include "src/Dense.h"
#include "src/Node.h"
#include "src/Activation.h"
#include "src/LossFunction.h"
#include <iostream>
#include <vector>
#include <cmath>

namespace neural_autodiff {

class XORSolver {
public:
    Dense network;
    std::vector<Matrix> training_inputs;
    std::vector<Matrix> training_targets;
    double learning_rate;

    void prepare_training_data() {
        training_inputs = {
            Matrix(2, 1, {0, 0}),
            Matrix(2, 1, {0, 1}),
            Matrix(2, 1, {1, 0}),
            Matrix(2, 1, {1, 1})
        };

        training_targets = {
            Matrix(1, 1, {0}),
            Matrix(1, 1, {1}),
            Matrix(1, 1, {1}),
            Matrix(1, 1, {0})
        };
    }

    void backpropagate(NodePtr output, NodePtr target) {
        // Compute initial gradient (derivative of MSE)
        Matrix gradient(output->value_.rows, output->value_.cols);
        for (int i = 0; i < output->value_.rows; ++i) {
            for (int j = 0; j < output->value_.cols; ++j) {
                gradient.at(i, j) = 2 * (output->value_.at(i, j) - target->value_.at(i, j));
            }
        }

        // Update weights manually
        if (network.layers_.size() > 0) {
            auto layers = network.layers_;
            auto last_layer = layers.back();

            for (int i = 0; i < last_layer->weights_->value_.rows; ++i) {
                for (int j = 0; j < last_layer->weights_->value_.cols; ++j) {
                    last_layer->weights_->value_.at(i, j) -=
                        learning_rate * gradient.at(0, 0) * output->inputs_[0]->value_.at(j, 0);
                }

                // Update bias
                last_layer->bias_->value_.at(i, 0) -=
                    learning_rate * gradient.at(0, 0);
            }
        }
    }

public:
    XORSolver(double lr = 0.1) : learning_rate(lr) {
        // Configure network architecture
        network.add_layer(2, 4, Activation::relu);
        network.add_layer(4, 1);

        prepare_training_data();
    }

    void train(int epochs) {
        for (int epoch = 0; epoch < epochs; ++epoch) {
            double total_loss = 0.0;

            for (int i = 0; i < training_inputs.size(); ++i) {
                // Forward pass
                NodePtr input = Node::make_input(training_inputs[i]);
                NodePtr target = Node::make_input(training_targets[i]);
                NodePtr output = network.forward(input);

                NodePtr loss = Loss::mse_loss(output, target);
                total_loss += loss->value_.at(0, 0);

                backpropagate(output, target);

                // Reset gradients
                network.zero_grad();
            }

            std::cout << "Epoch " << epoch<< " Average Loss: " << total_loss / training_inputs.size()
            << std::endl;

        }
    }

    void test() {
        std::cout << "XOR Test Results:\n";
        for (int i = 0; i < training_inputs.size(); ++i) {
            NodePtr input = Node::make_input(training_inputs[i]);
            NodePtr output = network.forward(input);

            std::cout << training_inputs[i].at(0,0) << " XOR "
                      << training_inputs[i].at(1,0) << " = "
                      << output->value_.at(0,0)
                      << " (Target: " << training_targets[i].at(0,0) << ")\n";
        }
    }
};

}

int main() {
    neural_autodiff::XORSolver solver;
    solver.train(20);
    solver.test();
    return 0;
}