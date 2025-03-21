#include "src/Layer.h"
#include "src/Dense.h"
#include "src/Node.h"
#include "src/Activation.h"
#include "src/LossFunction.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <set>

namespace neural_autodiff {

    std::set<char> vocab;
    int matrix[26][27];
    double probability[26][27];

    std::vector<Matrix> training_input, training_output;
    Dense network;
    double learning_rate = 0.01;

    std::vector<std::vector<char>> prediction(26);

    int charToIndex(char c) {
        if (c>='a' and c<='z') {
            return c-'a';
        }
        if (c>='A' and c<='Z') {
            return c-'A';
        }

        return 26;
    }

    void build_vocab(std::string str) {

        for (int i = 0; i < str.size()-1; ++i) {

            if (charToIndex(str[i]) > 25) {
                std::cout << str[i]<<" ";
                continue;
            }
            //std::cout << str[i]<<std::endl;
            matrix[charToIndex(str[i])][charToIndex(str[i+1])] += 1;
        }

    }

    void build_probabilities() {
        for (int i = 0; i < 26; ++i) {
            int sum = 0;
            for (int j = 0; j < 27; ++j) {
                sum += matrix[i][j];
            }

            for (int j = 0; j < 27; ++j) {
                if (sum!=0)
                    probability[i][j] = (double)matrix[i][j]/sum;
                else
                    probability[i][j] = 0;
            }
        }
    }

    std::vector<float> encode(char c) {
        std::vector<float> result(27);
        for (int i = 0; i < 27; ++i) {
            result[i] = i==charToIndex(c) ? 1 : 0;
        }

        return result;
    }

    void file_read()
    {
        std::string str = "", temp = "";

        std::ifstream read("/home/musfiq/CLionProjects/SPL-1/src/corpus.txt", std::ios::in | std::ios::binary);
        std::string process_string = "";

        if (!read.is_open())
        {
            std::cerr << "Error opening file" << std::endl;
        }
        while (std::getline(read, temp)) {
            //std::cout << temp << std::endl;
            for (char c: temp) {
                if (std::isalpha(c))
                {
                    process_string.push_back(std::tolower(c));
                }
                else {
                    process_string.push_back('.');
                }

            }
            str += process_string;
            temp = "";
        }

        build_vocab(process_string);
        read.close();
    }

    void prepare_train_set() {

        for (int i = 0; i < 26; ++i) {
            std::vector<double> mat(27);

            for (int j = 0; j < 27; ++j) {
                mat[j] = i==j? 1.0 : 0.0;
            }

            training_input.push_back(Matrix(27, 1, mat));

        }

        for (int i = 0; i < 27; ++i) {
            std::vector<double> mat(27);

            for (int j = 0; j < 27; ++j) {
                mat[j] = probability[i][j];
            }

            training_output.push_back(Matrix(27, 1, mat));
        }

    }

    void backpropagate(NodePtr output, NodePtr target) {
        // Compute initial gradient (derivative of MSE)
        Matrix gradient(output->value_.rows, output->value_.cols);
        for (int i = 0; i < output->value_.rows; ++i) {
            for (int j = 0; j < output->value_.cols; ++j) {
                gradient.at(i, j) = 2 * (output->value_.at(i, j) - target->value_.at(i, j));
            }
        }

        // Update weights manually
        if (network.layers_.size() > 0) {
            auto layers = network.layers_;
            auto last_layer = layers.back();

            for (int i = 0; i < last_layer->weights_->value_.rows; ++i) {
                for (int j = 0; j < last_layer->weights_->value_.cols; ++j) {
                    last_layer->weights_->value_.at(i, j) -=
                        learning_rate * gradient.at(0, 0) * output->inputs_[0]->value_.at(j, 0);
                }

                // Update bias
                last_layer->bias_->value_.at(i, 0) -=
                    learning_rate * gradient.at(0, 0);
            }
        }
    }

    void train(int epochs) {

        file_read();
        build_probabilities();

        prepare_train_set();

        for (int i=0; i<26; ++i)
        {
            for (int j=0; j<27; ++j)
            {
                std::cout << training_output[i].at(j, 0) << " ";
            }
            std::cout << std::endl;
        }

        for (int epoch = 0; epoch < epochs; ++epoch) {

            for (int i = 0; i < training_input.size(); ++i) {
                // Forward pass
                NodePtr input = Node::make_input(training_input[i]);
                NodePtr target = Node::make_input(training_output[i]);
                NodePtr output = network.forward(input);

                NodePtr loss = Loss::mse_loss(output, target);

                backpropagate(output, target);

                // Reset gradients
                network.zero_grad();
            }

            //std::cout << "Epoch " << epoch<< " Average Loss: " << total_loss / training_input.size()
            //<< std::endl;


        }

        std::cout << std::endl;
        std::cout << std::endl;
        for (int i=0; i<26; ++i)
        {
            for (int j=0; j<27; ++j)
            {
                std::cout << training_output[i].at(j, 0) << " ";
            }
            std::cout << std::endl;
        }

    }

    void prepare_predict()
    {

        for (int i = 0; i < 26; ++i)
        {
            std::vector<char> predic;

            for (int j = 0; j<27; ++j)
            {
                int k = training_output[i].at(j, 0)*100;
                while (k--)
                {
                    predic.push_back(j<26?(j+'a'):'.');
                }
            }

            while (predic.size()<100)
            {
                predic.push_back('.');
            }

            prediction[i] = predic;
        }

    }

    std::string predict(char c)
    {
        srand(time(0));
        std::string predicted_string = "";

        while (c != '.')
        {
            predicted_string.push_back(c);
            c = prediction[c-'a'][rand()%100];

        }

        return predicted_string;
    }

};

int main() {

    neural_autodiff::train(5);
    neural_autodiff::prepare_predict();

    std::cout << neural_autodiff::predict('e') << std::endl;
}