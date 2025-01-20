#include "src/Node.h"
#include "src/Layer.h"
#include "src/Activation.h"
#include "src/LossFunction.h"
#include "src/Gradient_Descent.h"
#include "src/Backward.h"
#include <iostream>
#include <vector>
#include <random>

using namespace neural_autodiff;

double compute_accuracy(const std::vector<Matrix>& predictions, const std::vector<Matrix>& targets) {
    int correct = 0;
    for (size_t i = 0; i < predictions.size(); ++i) {
        bool pred = predictions[i].at(0, 0) >= 0.5;
        bool target = targets[i].at(0, 0) >= 0.5;
        if (pred == target) correct++;
    }
    return static_cast<double>(correct) / predictions.size();
}

int main() {

    return 0;
}