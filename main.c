#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>

#define N 8

void init_vector(float vec[], size_t size) {
    for (size_t i = 0; i < size; i++) {
        vec[i] = (float)rand() / RAND_MAX;
    }
}

void init_matrix(float mat[][N], size_t size) {
    for (size_t i = 0; i < size; i++) {
        for (size_t j = 0; j < size; j++) {
            mat[i][j] = (float)rand() / RAND_MAX;
        }
    }
}

void print_vector(float vec[], size_t size) {
    printf("[");
    for (size_t i = 0; i < size; i++) {
        printf("%.2f ", vec[i]);
    }
    printf("]\n");
}

void print_matrix(float mat[][N], size_t size) {
    for (size_t i = 0; i < size; i++) {
        printf("[");
        for (size_t j = 0; j < size; j++) {
            printf("%.2f ", mat[i][j]);
        }
        printf("]\n");
    }
}

// c = c + M * b
static void simple_dgmv( size_t n , float c[ n ], const float M[n][n], const float b[n]) {
    for ( int i = 0; i < n ; i ++){
        for ( int j = 0; j < n ; j ++){
            c[i] += M[i][j]*b[j];
        }
    }
}

// c = c + M * b


int main() {
    srand(2384097);

    float b[N], c[N];
    float M[N][N];

    init_vector(b, N);
    init_vector(c, N);
    init_matrix(M, N);

    printf("Vector b:\n");
    print_vector(b, N);

    printf("\nVector c:\n");
    print_vector(c, N);

    printf("\nMatriz M:\n");
    print_matrix(M, N);

    simple_dgmv(N, c,M,b);

    printf("\nVector c (resultado):\n");
    print_vector(c, N);

}