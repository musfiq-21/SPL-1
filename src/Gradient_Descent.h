//
// Created by musfiq on 1/19/25.
//

#ifndef OPTIMIZER_H
#define OPTIMIZER_H
#include "Node.h"

namespace neural_autodiff {

    class Optimizer {
    public:
        Optimizer(const std::vector<NodePtr>& parameters, double learning_rate);
        virtual void step() = 0;
        virtual void zero_grad();

    protected:
        std::vector<NodePtr> parameters_;
        double learning_rate_;
    };

    class Gradient_Descent : public Optimizer {
    public:
        Gradient_Descent(const std::vector<NodePtr>& parameters,
            double learning_rate,
            double momentum = 0.9);
        void step() override;

    private:
        double momentum_;
        std::vector<std::vector<double>> velocities_;
    };

}
#endif //OPTIMIZER_H
