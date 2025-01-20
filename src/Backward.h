
#ifndef BACKWARD_H
#define BACKWARD_H
#include "Node.h"
namespace neural_autodiff {
    void backward(NodePtr node, const Matrix& gradient = Matrix(1, 1, {1.0}));
}

#endif //BACKWARD_H
