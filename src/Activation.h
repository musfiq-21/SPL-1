
#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "Node.h"

namespace neural_autodiff {

    class Activation {
    public:
        static NodePtr relu(NodePtr x);
        static NodePtr sigmoid(NodePtr x);
        static NodePtr tanh(NodePtr x);
    };

}
#endif //ACTIVATION_H
