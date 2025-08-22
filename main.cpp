#include <iostream>
#include <vector>
#include <algorithm>
#include "headers/Network.hpp"
#include "headers/utils.hpp"

using namespace std;

int main()
{
    try
    {
        // Load MNIST IDX files
        auto train_images = load_images("train-images.idx3-ubyte");
        auto train_labels_raw = load_labels("train-labels.idx1-ubyte");
        auto test_images = load_images("t10k-images.idx3-ubyte");
        auto test_labels_raw = load_labels("t10k-labels.idx1-ubyte");

        // Convert labels to one-hot vectors
        vector<vector<double>> train_labels;
        for (int l : train_labels_raw)
            train_labels.push_back(one_hot(l));

        vector<vector<double>> test_labels;
        for (int l : test_labels_raw)
            test_labels.push_back(one_hot(l));

        // Optional: use smaller subset for faster testing
        int n_train = 10000;
        int n_test = 2000;

        vector<vector<double>> train_x(train_images.begin(), train_images.begin() + n_train);
        vector<vector<double>> train_y(train_labels.begin(), train_labels.begin() + n_train);

        vector<vector<double>> test_x(test_images.begin(), test_images.begin() + n_test);
        vector<vector<double>> test_y(test_labels.begin(), test_labels.begin() + n_test);

        // Prepare training and test data in pair format
        vector<pair<vector<double>, vector<double>>> training_data;
        for (int i = 0; i < n_train; i++)
            training_data.push_back({train_x[i], train_y[i]});

        vector<pair<vector<double>, vector<double>>> test_data;
        for (int i = 0; i < n_test; i++)
            test_data.push_back({test_x[i], test_y[i]});

        // Building and training the network
        Network net({784, 128, 64, 10});                // input, { hidden }, output
        net.SGD(training_data, 30, 3, 3.0, test_data); // {epochs = 30, batch = 3, eta=3.0}

        // Evaluate on full test set
        int correct = 0;
        for (size_t i = 0; i < test_x.size(); i++)
        {
            auto output = net.feedforward(test_x[i]);
            int predicted = (int)(max_element(output.begin(), output.end()) - output.begin());
            int actual = (int)(max_element(test_y[i].begin(), test_y[i].end()) - test_y[i].begin());
            if (predicted == actual)
                correct++;
        }

        cout << "Test accuracy: " << (100.0 * correct / test_x.size()) << "%\n";
    }
    catch (const exception &ex)
    {
        cerr << "Error: " << ex.what() << "\n";
        return 1;
    }

    return 0;
}
