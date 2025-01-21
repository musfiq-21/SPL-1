
#ifndef LAYER_H
#define LAYER_H

#include "Node.h"

namespace neural_autodiff {

    class Layer {
    public:
        virtual ~Layer() = default;
        virtual NodePtr forward(NodePtr input) = 0;
        virtual std::vector<NodePtr> parameters() const = 0;
        virtual void zero_grad() = 0;
    };

    class LinearLayer : public Layer {
    public:
        LinearLayer(int in_features, int out_features);
        NodePtr forward(NodePtr input) override;
        std::vector<NodePtr> parameters() const override;
        void zero_grad() override;

    private:
        NodePtr weights_;
        NodePtr bias_;
        size_t in_features_;
        size_t out_features_;
    };

}
#endif //LAYER_H
