#ifndef MATRIX_H
#define MATRIX_H

#include <vector>

struct Matrix {
    int r, c;
    std::vector<double> d;
    Matrix(int r = 0, int c = 0, double v = 0) : r(r), c(c), d(r * c, v) {}
    double& operator()(int i, int j) { return d[i * c + j]; }
};

Matrix transpose(Matrix& A);
Matrix dot(Matrix& A, Matrix& B);

#endif // MATRIX_H
