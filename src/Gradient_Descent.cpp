#include "Gradient_Descent.h"
#include <cmath>
#include <stdexcept>

namespace neural_autodiff {
    Optimizer::Optimizer(const std::vector<NodePtr>& parameters, double learning_rate)
        : parameters_(parameters)
        , learning_rate_(learning_rate) {
        if (learning_rate <= 0) {
            throw std::invalid_argument("Learning rate must be positive");
        }
    }

    void Optimizer::zero_grad() {
        for (auto& param : parameters_) {
            param->zero_grad();
        }
    }

    Gradient_Descent::Gradient_Descent(const std::vector<NodePtr>& parameters, double learning_rate, double momentum)
        : Optimizer(parameters, learning_rate)
        , momentum_(momentum) {
        if (momentum < 0 or momentum >= 1) {
            throw std::invalid_argument("Momentum must be in [0, 1)");
        }

        velocities_.resize(parameters.size());
        for (int i = 0; i < parameters.size(); ++i) {
            velocities_[i].resize(parameters[i]->value_.rows * parameters[i]->value_.cols, 0.0);
        }
    }

    void Gradient_Descent::step() {
        for (int i = 0; i < parameters_.size(); ++i) {
            auto& param = parameters_[i];
            auto& velocity = velocities_[i];

            int idx = 0;
            for (int row = 0; row < param->value_.rows; ++row) {
                for (int col = 0; col < param->value_.cols; ++col) {
                    // Update velocity
                    velocity[idx] = momentum_ * velocity[idx] + learning_rate_ * param->grad_.at(row, col);

                    // Update parameter
                    param->value_.at(row, col) -= velocity[idx];
                    idx++;
                }
            }
        }
    }
}
