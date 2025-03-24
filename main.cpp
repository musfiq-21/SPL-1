#include "src/Layer.h"
#include "src/Dense.h"
#include "src/Node.h"
#include "src/Matrix.h"
#include "src/LossFunction.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <map>
#include <set>

namespace neural_autodiff {

    std::set<char> vocab;
    int matrix[26][27];
    double probability[26][27];
    std::vector<Matrix> prediction_probs;


    std::vector<std::vector<std::string>> vocabulary;
    std::map<std::string, int> vocab_map;

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

    char indexToChar(int k)
    {
        if (0<=k and k<26)
        {
            return k+'a';
        }
        return '.';
    }
    void build_vocab(std::string str) {

        for (int i = 0; i < str.size()-1; ++i) {

            if (charToIndex(str[i]) > 25) {
                std::cout << str[i]<<" ";
                continue;
            }

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
                mat[j] = (i == j) ? 1.0 : 0.0;
            }
            training_input.push_back(Matrix(27, 1, mat));
        }
        for (int i = 0; i < 26; ++i) { // Change to 26
            std::vector<double> mat(27);
            for (int j = 0; j < 27; ++j) {
                mat[j] = probability[i][j];
            }
            training_output.push_back(Matrix(27, 1, mat));
        }
    }

    void backpropagate(NodePtr output, NodePtr target) {
        Matrix gradient(output->value_.rows, output->value_.cols);
        for (int i = 0; i < output->value_.rows; ++i) {
            for (int j = 0; j < output->value_.cols; ++j) {
                gradient.at(i, j) = 2 * (output->value_.at(i, j) - target->value_.at(i, j));
            }
        }

        if (network.layers_.size() > 0) {
            auto last_layer = network.layers_.back();
            for (int i = 0; i < last_layer->weights_->value_.rows; ++i) {
                for (int j = 0; j < last_layer->weights_->value_.cols; ++j) {
                    last_layer->weights_->value_.at(i, j) -=
                        learning_rate * gradient.at(i, 0) * output->inputs_[0]->value_.at(j, 0);
                }
                last_layer->bias_->value_.at(i, 0) -= learning_rate * gradient.at(i, 0);
            }
        }
    }

    void show_weights_and_biases(const std::string& label) {
        if (network.layers_.empty()) {
            std::cout << label << ": Network has no layers!" << std::endl;
            return;
        }
        auto layer = network.layers_[0]; // Assuming one layer
        std::cout << label << " - Layer 0:" << std::endl;

        double w_mean = 0.0, w_std = 0.0;
        int w_count = layer->weights_->value_.rows * layer->weights_->value_.cols;
        for (int i = 0; i < layer->weights_->value_.rows; ++i) {
            for (int j = 0; j < layer->weights_->value_.cols; ++j) {
                double w = layer->weights_->value_.at(i, j);
                w_mean += w;
                w_std += w * w;
            }
        }
        w_mean /= w_count;
        w_std = sqrt(w_std / w_count - w_mean * w_mean);
        std::cout << "  Weights - Mean: " << w_mean << ", Std: " << w_std << std::endl;


        double b_mean = 0.0;
        int b_count = layer->bias_->value_.rows;
        for (int i = 0; i < b_count; ++i) {
            b_mean += layer->bias_->value_.at(i, 0);
        }
        b_mean /= b_count;
        std::cout << "  Biases - Mean: " << b_mean << std::endl;
    }

    double compute_mse(const Matrix& output, const Matrix& target) {

        if (output.rows != target.rows || output.cols != target.cols) {
            throw std::invalid_argument("Output and target matrices must have the same dimensions");
        }


        double sum_squared_diff = 0.0;
        int num_elements = output.rows * output.cols;
        for (int i = 0; i < output.rows; ++i) {
            for (int j = 0; j < output.cols; ++j) {
                double diff = output.at(i, j) - target.at(i, j);
                sum_squared_diff += diff * diff;
            }
        }


        return sum_squared_diff / num_elements;
    }

    Matrix softmax(const Matrix& input) {
        if (input.cols != 1) {
            throw std::invalid_argument("Softmax expects a column vector");
        }
        Matrix result(input.rows, 1);
        // Subtract max for numerical stability
        double max_val = input.at(0, 0);
        for (int i = 1; i < input.rows; ++i) {
            if (input.at(i, 0) > max_val) max_val = input.at(i, 0);
        }
        double sum_exp = 0.0;
        for (int i = 0; i < input.rows; ++i) {
            result.at(i, 0) = exp(input.at(i, 0) - max_val);
            sum_exp += result.at(i, 0);
        }
        for (int i = 0; i < input.rows; ++i) {
            result.at(i, 0) /= sum_exp;
        }
        return result;
    }
    void train(int epochs) {
        file_read();
        build_probabilities();
        prepare_train_set();

        if (network.layers_.empty()) {
            network = Dense();
            std::vector<double> weights(27 * 27);
            for (auto& w : weights) w = ((double)rand() / RAND_MAX) * 0.1 - 0.05;
            Matrix weights_matrix = Matrix(27, 27, weights);
            std::vector<double> biases(27, 0.0);
            Matrix biases_matrix = Matrix(27, 1, biases);
            network.add_layer(27, 27, weights_matrix, biases_matrix);
            show_weights_and_biases("Before Training");
        }

        for (int e = 0; e < epochs; ++e) {
            for (int i = 0; i < training_input.size(); ++i) {
                NodePtr input = Node::make_input(training_input[i]);
                NodePtr target = Node::make_input(training_output[i]);
                NodePtr output = network.forward(input);
                backpropagate(output, target);
                network.zero_grad();
            }
            // Optional: Show after each epoch
            show_weights_and_biases("After Epoch " + std::to_string(e));

            // MSE monitoring
            double total_mse = 0.0;
            for (int i = 0; i < training_input.size(); ++i) {
                NodePtr input = Node::make_input(training_input[i]);
                NodePtr output = network.forward(input);
                total_mse += compute_mse(output->value_, training_output[i]);
            }
            double avg_mse = total_mse / training_input.size();
            std::cout << "Epoch " << e << ", Training MSE Loss: " << avg_mse << std::endl;

        }
    }
    // void prepare_predict() {
    //     for (int i = 0; i < 26; ++i) {
    //         NodePtr input = Node::make_input(training_input[i]);
    //         NodePtr output = network.forward(input);
    //         std::vector<char> predic;
    //         for (int j = 0; j < 27; ++j) {
    //             int k = output->value_.at(j, 0) * 100; // Use network output
    //             if (k < 0) k = 0; // Ensure non-negative for sampling
    //             while (k--) {
    //                 predic.push_back(j < 26 ? (j + 'a') : '.');
    //             }
    //         }
    //         while (predic.size() < 100) {
    //             predic.push_back('.');
    //         }
    //         prediction[i] = predic;
    //     }
    // }

    void prepare_predict() {
        for (int i = 0; i < 26; ++i) {
            Matrix input(27, 1); // Change from 26 to 27
            for (int j = 0; j < 27; ++j) {
                input.at(j, 0) = (i == j) ? 1.0 : 0.0;
            }
            NodePtr input_node = Node::make_input(input);
            NodePtr output = network.forward(input_node);
            Matrix prob = softmax(output->value_);
            prediction_probs.push_back(prob);
        }
    }

    char getChar(char c)
    {
        std::map<double, char> probs;

        for (int i=0; i<26; ++i)
        {
            probs.insert(std::pair<double, char>(-probability[c-'a'][i], c));
        }

        char k = prediction[c-'a'][rand()%100];

        return k;
    }
    std::string predict(char c) {
        srand(time(0));
        std::string predicted_string = "";

        while (c != '.') {
            predicted_string.push_back(c);
            int idx = c - 'a';
            if (idx < 0 || idx >= 26)
                break;

            Matrix prob = prediction_probs[idx];

            double r = (double)rand() / RAND_MAX; // Random value in [0,1]
            double cum_prob = 0.0;
            char next_c = '.';
            for (int j = 0; j < 27; ++j) {
                cum_prob += prob.at(j, 0);
                if (r <= cum_prob) {
                    next_c = (j < 26) ? (j + 'a') : '.';
                    break;
                }
            }
            c = next_c;
        }
        return predicted_string;
    }

    void show_probabilities()
    {
        printf("    ");
        for (int i = 0; i < 27; ++i)
        {
            printf("  %c    ", indexToChar(i));
        }
        printf("\n");

        for (int i=0; i<26; ++i)
        {
            printf("%c  ", indexToChar(i));
            for (int j=0; j<27; ++j)
            {
                printf(" %.3f ", probability[i][j]);
            }
            printf("\n");
        }
    }

    void show_updated_output()
    {
        printf("    ");
        for (int i = 0; i < 27; ++i)
        {
            printf("  %c    ", indexToChar(i));
        }
        printf("\n");

        for (int i=0; i<26; ++i)
        {
            printf("%c  ", indexToChar(i));
            for (int j=0; j<27; ++j)
            {
                printf(" %.3f ", training_output[i].at(j, 0));
                //printf("%f", );
            }
            printf("\n");
        }
    }


};

int main() {

    neural_autodiff::train(5);

    neural_autodiff::prepare_predict();
    std::cout<<std::endl;
    neural_autodiff::show_probabilities();
    std::cout<<std::endl;
    std::cout<<std::endl;
    //neural_autodiff::show_updated_output();
    int k;
    std::cout << "Bigram text generation, using Custom neural network library named 'Neurabuild'"<<std::endl;
    std::cout << "------------------------------------------------------------------------------"<<std::endl;
    std::cout << "Enter number of prompt: ";
    std::cin>>k;

    while (k--)
    {
        char c;
        std::cout << "Prompt: ";
        std::cin>>c;
        neural_autodiff::predict(c);

        std::cout <<"Generated text: ";
        std::cout<<neural_autodiff::predict(c)<<std::endl;
    }
}