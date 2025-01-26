
#ifndef OPTIMIZER_H
#define OPTIMIZER_H
#include "Node.h"

namespace neural_autodiff {

    class Gradient_Descent {
    public:
        Gradient_Descent(const std::vector<NodePtr>& parameters, double learning_rate);
        //void step();

    };

}
#endif //OPTIMIZER_H
