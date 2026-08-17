#include "Matrix.h"

using namespace std;

Matrix transpose(Matrix& A) {
    Matrix B(A.c, A.r);
    for (int i = 0; i < A.r; ++i)
        for (int j = 0; j < A.c; ++j) B(j, i) = A(i, j);
    return B;
}

Matrix dot(Matrix& A, Matrix& B) {
    Matrix C(A.r, B.c);
    for (int i = 0; i < A.r; ++i)
        for (int k = 0; k < A.c; ++k)
            for (int j = 0; j < B.c; ++j)
                C(i, j) += A(i, k) * B(k, j);
    return C;
}
