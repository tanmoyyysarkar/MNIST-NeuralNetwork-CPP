#include <iostream>
#include <vector>
#include <cmath>
#include "headers/utils.hpp"

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

int reverseInt(int i)
{
    unsigned char c1 = i & 255;
    unsigned char c2 = (i >> 8) & 255;
    unsigned char c3 = (i >> 16) & 255;
    unsigned char c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

vector<vector<double>> load_images(const string &filename)
{
    ifstream file(filename, ios::binary);
    if (!file.is_open())
        throw runtime_error("Could not open " + filename);

    int magic_number = 0, number_of_images = 0, n_rows = 0, n_cols = 0;
    file.read((char *)&magic_number, sizeof(magic_number));
    file.read((char *)&number_of_images, sizeof(number_of_images));
    file.read((char *)&n_rows, sizeof(n_rows));
    file.read((char *)&n_cols, sizeof(n_cols));

    magic_number = reverseInt(magic_number);
    number_of_images = reverseInt(number_of_images);
    n_rows = reverseInt(n_rows);
    n_cols = reverseInt(n_cols);

    vector<vector<double>> images(number_of_images, vector<double>(n_rows * n_cols));
    for (int i = 0; i < number_of_images; i++)
    {
        for (int r = 0; r < n_rows * n_cols; r++)
        {
            unsigned char temp = 0;
            file.read((char *)&temp, sizeof(temp));
            images[i][r] = ((double)temp) / 255.0;
        }
    }
    return images;
}

vector<int> load_labels(const string &filename)
{
    ifstream file(filename, ios::binary);
    if (!file.is_open())
        throw runtime_error("Could not open " + filename);

    int magic_number = 0, number_of_labels = 0;
    file.read((char *)&magic_number, sizeof(magic_number));
    file.read((char *)&number_of_labels, sizeof(number_of_labels));

    magic_number = reverseInt(magic_number);
    number_of_labels = reverseInt(number_of_labels);

    vector<int> labels(number_of_labels);
    for (int i = 0; i < number_of_labels; i++)
    {
        unsigned char temp = 0;
        file.read((char *)&temp, sizeof(temp));
        labels[i] = (int)temp;
    }
    return labels;
}

// One-hot encoding
vector<double> one_hot(int label, int num_classes)
{
    vector<double> vec(num_classes, 0.0);
    vec[label] = 1.0;
    return vec;
}
