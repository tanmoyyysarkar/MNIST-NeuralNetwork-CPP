#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>

using namespace std;

// dot product of matrix(M x N) x vector (N) -> vector (M)
vector<double> dot(const vector<vector<double>> &mat, const vector<double> &vec);

// transpose outer product: vector(M) * vector(N)^T -> matrix(MxN)
vector<vector<double>> outer(const vector<double> &a, const vector<double> &b);

int reverseInt(int i);

vector<vector<double>> load_images(const string &filename);

vector<int> load_labels(const string &filename);

// One-hot encoding
vector<double> one_hot(int label, int num_classes = 10);
