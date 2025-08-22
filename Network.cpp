#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <headers/Network.hpp>

using namespace std;

Network::Network(vector<int> sizes)
{
    this->num_layers = sizes.size();
    this->sizes = sizes;

    random_device rd;
    mt19937 gen(rd());
    normal_distribution<> dist(0.0, 1.0);

    for (int i = 1; i < num_layers; i++)
    {
        vector<double> bias_vec(sizes[i]);
        for (int j = 0; j < sizes[i]; j++)
        {
            bias_vec[j] = dist(gen);
        }
        biases.push_back(bias_vec);
    }

    for (int i = 0; i < num_layers - 1; i++)
    {
        int x = sizes[i];
        int y = sizes[i + 1];

        vector<vector<double>> weight_matrix(y, vector<double>(x));
        for (int row = 0; row < y; row++)
        {
            for (int col = 0; col < x; col++)
            {
                weight_matrix[row][col] = dist(gen);
            }
        }
        weights.push_back(weight_matrix);
    }
}

void Network::SGD(vector<pair<vector<double>, vector<double>>> training_data, int epochs, int mini_batch_size, double eta, vector<pair<vector<double>, vector<double>>> test_data = {})
{
    int n = training_data.size();
    int n_test = test_data.size();

    for (int j = 0; j < epochs; j++)
    {
        shuffle(training_data.begin(), training_data.end(), mt19937(random_device()())); // shuffling the training data, needs the <algorithm> header file

        for (int k = 0; k < n; k += mini_batch_size)
        {
            int end = min(k + mini_batch_size, n);
            vector<pair<vector<double>, vector<double>>> mini_batch(
                training_data.begin() + k,
                training_data.begin() + end);

            update_mini_batch(mini_batch, eta);
        }
        if (!test_data.empty())
        {
            cout << "Epoch " << j << ": " << evaluate(test_data) << " / " << n_test << endl;
        }
        else
        {
            cout << "Epoch " << j << " complete" << endl;
        }
    }
}

// Evaluate function
int Network::evaluate(const vector<pair<vector<double>, vector<double>>> &test_data)
{
    int correct = 0;
    for (int i = 0; i < test_data.size(); i++)
    {
        const vector<double> x = test_data[i].first;
        const vector<double> &y = test_data[i].second;

        vector<double> output = feedforward(x);

        int prediction = 0;
        double max_val = output[0];

        for (int j = 1; j < output.size(); j++)
        {
            if (output[j] > max_val)
            {
                max_val = output[j];
                prediction = j;
            }
        }

        int actual = 0;
        double max_y = y[0];
        for (int j = 0; j < y.size(); j++)
        {
            if (y[j] > max_y)
            {
                max_y = y[j];
                actual = j;
            }
        }

        if (prediction == actual)
        {
            correct++;
        }
    }
    return correct;
}

vector<double> Network::feedforward(const vector<double> &input)
{
    vector<double> a = input;

    for (int l = 0; l < weights.size(); l++)
    {
        vector<double> z(biases[l].size());

        // z = w . a + b
        for (int i = 0; i < weights[l].size(); i++)
        {
            double sum = biases[l][i]; // added b in the sum
            for (int j = 0; j < weights[l][i].size(); j++)
            {
                sum += weights[l][i][j] * a[j];
            }
            z[i] = sum;
        }
        a = sigmoid_vec(z);
    }
    return a;
}

void Network::update_mini_batch(const vector<pair<vector<double>, vector<double>>> &mini_batch, double eta)
{
    // initializing the gradients
    vector<vector<double>> nabla_b(biases.size());
    vector<vector<vector<double>>> nabla_w(weights.size());

    // making biases and weights 0
    for (int i = 0; i < biases.size(); i++)
    {
        nabla_b[i].assign(biases[i].size(), 0.0);
    }

    for (int i = 0; i < weights.size(); i++)
    {
        nabla_w[i].resize(weights[i].size());
        for (int j = 0; j < weights[i].size(); j++)
        {
            nabla_w[i][j].assign(weights[i][j].size(), 0.0);
        }
    }

    // looping through each (x, y) in mini_batch
    for (int idx = 0; idx < (int)mini_batch.size(); idx++)
    {
        const vector<double> &x = mini_batch[idx].first;
        const vector<double> &y = mini_batch[idx].second;

        pair<vector<vector<double>>, vector<vector<vector<double>>>> result = backprop(x, y);

        vector<vector<double>> delta_nabla_b = result.first;
        vector<vector<vector<double>>> delta_nabla_w = result.second;

        // Accumulate Gradients
        for (int i = 0; i < nabla_b.size(); i++)
        {
            for (int j = 0; j < nabla_b[i].size(); j++)
            {
                nabla_b[i][j] += delta_nabla_b[i][j];
            }
        }
        for (int i = 0; i < nabla_w.size(); i++)
        {
            for (int j = 0; j < nabla_w[i].size(); j++)
            {
                for (int k = 0; k < nabla_w[i][j].size(); k++)
                {
                    nabla_w[i][j][k] += delta_nabla_w[i][j][k];
                }
            }
        }
    }
    // Updating weights and biases
    for (int i = 0; i < nabla_b.size(); i++)
    {
        for (int j = 0; j < nabla_b[i].size(); j++)
        {
            biases[i][j] -= (eta / mini_batch.size()) * nabla_b[i][j];
        }
    }
    for (int i = 0; i < nabla_w.size(); i++)
    {
        for (int j = 0; j < nabla_w[i].size(); j++)
        {
            for (int k = 0; k < nabla_w[i][j].size(); k++)
            {
                weights[i][j][k] -= (eta / mini_batch.size()) * nabla_w[i][j][k];
            }
        }
    }
}

// Back Propagation Function
pair<vector<vector<double>>, vector<vector<vector<double>>>> Network::backprop(const vector<double> &x, const vector<double> &y)
{
    // initialize nabla_b and nabla_w with zereos, size is identical to biases and weights
    vector<vector<double>> nabla_b;
    vector<vector<vector<double>>> nabla_w;

    for (int i = 0; i < biases.size(); i++)
    {
        nabla_b.push_back(vector<double>(biases[i].size(), 0.0));
    }
    for (int i = 0; i < weights.size(); i++)
    {
        nabla_w.push_back(vector<vector<double>>(weights[i].size(), vector<double>(weights[i][0].size(), 0.0)));
    }

    // feed Forward
    vector<double> activation = x;
    vector<vector<double>> activations; // stores all activations

    activations.push_back(x);

    vector<vector<double>> zs; // stores all z vectors

    for (int i = 0; i < biases.size(); i++)
    {
        vector<double> z = dot(weights[i], activation);
        for (int j = 0; j < z.size(); j++)
        {
            z[j] += biases[i][j];
        }
        zs.push_back(z);

        for (int j = 0; j < z.size(); j++)
        {
            z[j] = sigmoid(z[j]);
        }
        activation = z;
        activations.push_back(activation);
    }

    // back pass
    vector<double> delta = cost_derivative(activations.back(), y);
    for (int i = 0; i < delta.size(); i++)
    {
        delta[i] *= sigmoid_prime(zs.back()[i]);
    }
    nabla_b.back() = delta;
    nabla_w.back() = outer(delta, activations[activations.size() - 2]);

    // loop over earlier layers
    for (int l = 2; l < (int)biases.size(); l++)
    {
        vector<double> z = zs[zs.size() - 1];
        vector<double> sp(z.size(), 0.0);
        for (int j = 0; j < z.size(); j++)
        {
            sp[j] = sigmoid_prime(z[j]);
        }

        // transpose(weights[-l+1]) * delta
        vector<double> new_delta(weights[weights.size() - l + 1][0].size(), 0.0);
        for (int j = 0; j < new_delta.size(); j++)
        {
            for (int k = 0; k < delta.size(); k++)
            {
                new_delta[j] += weights[weights.size() - l + 1][k][j] * delta[k];
            }
            new_delta[j] *= sp[j];
        }
        delta = new_delta;
        nabla_b[nabla_b.size() - l] = delta;
        nabla_w[nabla_w.size() - l] = outer(delta, activations[activations.size() - l - 1]);
    }
    return make_pair(nabla_b, nabla_w);
}
