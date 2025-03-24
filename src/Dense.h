#include <functional>

#include "Layer.h"
#include "Activation.h"
#include <vector>
#include <memory>
#include <unordered_set>

namespace neural_autodiff {

    class Dense {

    public:
        std::vector<std::shared_ptr<LinearLayer>> layers_;
        std::vector<std::function<NodePtr(NodePtr)>> activations_;

        Dense() = default;

        void add_layer(int in_features, int out_features,
               const Matrix& weights,
               const Matrix& biases) {
            // Check weights dimensions: should be out_features x in_features
            if (weights.rows != out_features || weights.cols != in_features) {
                throw std::invalid_argument("Weights dimensions do not match");
            }
            // Check biases dimensions: should be out_features x 1
            if (biases.rows != out_features || biases.cols != 1) {
                throw std::invalid_argument("Biases dimensions do not match");
            }
            // Check compatibility with the previous layer (if any)
            if (!layers_.empty()) {
                int prev_out_features = layers_.back()->out_features_;
                if (prev_out_features != in_features) {
                    throw std::invalid_argument("Layer dimensions do not match");
                }
            }
            // Create and add the new LinearLayer with the provided weights and biases
            layers_.push_back(std::make_shared<LinearLayer>(in_features, out_features, weights, biases));
        }

        NodePtr forward(NodePtr input) {
            if (!input) {
                throw std::runtime_error("Input is null");
            }
            NodePtr current = input;
            for (size_t i = 0; i < layers_.size(); ++i) {
                if (!layers_[i]) {
                    throw std::runtime_error("Layer at index " + std::to_string(i) + " is null");
                }
                current = layers_[i]->forward(current);
                // if (activations_[i]) {
                //     current = activations_[i](current);
                // }
            }
            return current;
        }
        std::vector<NodePtr> get_nodes_in_topological_order(NodePtr output) {
            std::vector<NodePtr> result;
            std::unordered_set<NodePtr> visited;

            // Helper function for depth-first traversal
            std::function<void(NodePtr)> dfs = [&](NodePtr node) {
                // Skip if null or already visited
                if (!node || visited.find(node) != visited.end()) {
                    return;
                }

                visited.insert(node);

                // Visit all input nodes first (parents/dependencies)
                for (const auto& input : node->inputs_) {
                    dfs(input);
                }

                // After all dependencies are processed, add this node
                result.push_back(node);
            };

            // Start DFS from the output node
            dfs(output);
            return result;
        }
        void zero_grad() {
            for (auto& layer : layers_) {
                layer->zero_grad();
            }
        }

    };

}
