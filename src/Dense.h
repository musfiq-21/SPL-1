#include <functional>

#include "Layer.h"
#include "Activation.h"
#include <vector>
#include <memory>

namespace neural_autodiff {

    class Dense {

    public:
        std::vector<std::shared_ptr<LinearLayer>> layers_;
        std::vector<std::function<NodePtr(NodePtr)>> activations_;

        Dense() = default;

        void add_layer(int in_features, int out_features,
                       std::function<NodePtr(NodePtr)> activation = nullptr) {
            if (!layers_.empty()) {

                int prev_out_features = layers_.back()->out_features_;
                if (prev_out_features != in_features) {
                    throw std::invalid_argument("Layer dimensions do not match");
                }
            }

            layers_.push_back(std::make_shared<LinearLayer>(in_features, out_features));
            activations_.push_back(activation);
        }

        NodePtr forward(NodePtr input) {
            NodePtr current = input;
            for (size_t i = 0; i < layers_.size(); ++i) {
                current = layers_[i]->forward(current);

                if (activations_[i]) {
                    current = activations_[i](current);
                }
            }
            return current;
        }

        void zero_grad() {
            for (auto& layer : layers_) {
                layer->zero_grad();
            }
        }

    };

}