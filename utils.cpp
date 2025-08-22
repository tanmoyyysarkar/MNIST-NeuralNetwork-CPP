#include <iostream>
#include <vector>
#include <cmath>
#include <headers/utils.hpp>

using namespace std;




// dot product of matrix(M x N) x vector (N) -> vector (M)
vector<double> dot(const vector<vector<double>> &mat, const vector<double> &vec)
{
    vector<double> res(mat.size(), 0.0);
    for (int i = 0; i < mat.size(); i++)
    {
        for (int j = 0; j < mat[0].size(); j++)
        {
            res[i] += mat[i][j] * vec[j];
        }
    }
    return res;
}

// transpose outer product: vector(M) * vector(N)^T -> matrix(MxN)
vector<vector<double>> outer(const vector<double> &a, const vector<double> &b)
{
    vector<vector<double>> result(a.size(), vector<double>(b.size(), 0.0));
    for (int i = 0; i < a.size(); i++)
    {
        for (int j = 0; j < b.size(); j++)
        {
            result[i][j] = a[i] * b[j];
        }
    }
    return result;
}




