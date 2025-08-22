#include <iostream>
#include <vector>
#include <cmath>

using namespace std;



// dot product of matrix(M x N) x vector (N) -> vector (M)
vector<double> dot(const vector<vector<double>> &mat, const vector<double> &vec);

// transpose outer product: vector(M) * vector(N)^T -> matrix(MxN)
vector<vector<double>> outer(const vector<double> &a, const vector<double> &b);


