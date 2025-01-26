
#ifndef LOSSFUNCTION_H
#define LOSSFUNCTION_H
#include "Node.h"

namespace neural_autodiff {

    class Loss {
    public:
        static NodePtr mse_loss(NodePtr predicted, NodePtr target);
        static NodePtr maximum_log_likelihood(NodePtr predicted, NodePtr target);

        static NodePtr mse_loss_prime(NodePtr predicted, NodePtr target);
    };

}
#endif //LOSSFUNCTION_H
