#include <iostream>
#include <vector>
#include <cmath>
using namespace std;

class Network
{
private:
    int num_layers;
    vector<int> sizes;
    vector<vector<double>> biases;
    vector<vector<vector<double>>> weights;

public:
    Network(vector<int> sizes);

    void SGD(vector<pair<vector<double>, vector<double>>> training_data, int epochs, int mini_batch_size, double eta, vector<pair<vector<double>, vector<double>>> test_data = {});

private:
    // dot product of matrix(M x N) x vector (N) -> vector (M)
    vector<double> dot(const vector<vector<double>> &mat, const vector<double> &vec);

    // transpose outer product: vector(M) * vector(N)^T -> matrix(MxN)
    vector<vector<double>> outer(const vector<double> &a, const vector<double> &b);

    // cost derivative
    vector<double> cost_derivative(const vector<double> &output, const vector<double> &y);

    // Evaluate function
    int evaluate(const vector<pair<vector<double>, vector<double>>> &test_data);

    vector<double> sigmoid_vec(const vector<double> &z);

    vector<double> feedforward(const vector<double> &input);

    // calculates the sigmoid vector
    vector<double> sigmoid_vec(const vector<double> &z);

    void update_mini_batch(const vector<pair<vector<double>, vector<double>>> &mini_batch, double eta);

    // Back Propagation Function
    pair<vector<vector<double>>, vector<vector<vector<double>>>> backprop(const vector<double> &x, const vector<double> &y);
};
