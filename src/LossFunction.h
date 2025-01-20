
#ifndef LOSSFUNCTION_H
#define LOSSFUNCTION_H
#include "Node.h"

namespace neural_autodiff {

    class Loss {
    public:
        static NodePtr mse_loss(NodePtr predicted, NodePtr target);

    };

}
#endif //LOSSFUNCTION_H
