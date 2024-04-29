#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N 8

void init_vector(float vec[], size_t size) {
    for (size_t i = 0; i < size; i++) {
        vec[i] = (float)(rand() % 10);
    }
}

void init_matrix(float mat[][N], size_t size) {
    for (size_t i = 0; i < size; i++) {
        for (size_t j = 0; j < size; j++) {
            mat[i][j] = (float)(rand() % 10);
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
static void unroll_dgmv( size_t n, float c[n], const float M[n][n], const float b[n]) {
    for (int i = 0 ; i<n ; i++)
        for (int j = 0 ; j<n ; j += 4)
            c[i] += M[i][j+0]*b[j+0]+
                    M[i][j+1]*b[j+1]+
                    M[i][j+2]*b[j+2]+
                    M[i][j+3]*b[j+3];
}

// c = c + M * b
static void avx512_dgmv( size_t n, float c[n], const float M[n][n], const float b[n]) {
    for (int i = 0 ; i<n ; i++)
        for (int j = 0 ; j<n ; j += 4)
            c[i] += M[i][j+0]*b[j+0]+
                    M[i][j+1]*b[j+1]+
                    M[i][j+2]*b[j+2]+
                    M[i][j+3]*b[j+3];
}


int main() {
    srand(2384097);

    float b[N], c[N], res1[N], res2[N],res3[N];
    float M[N][N];

    init_vector(c, N);
    init_vector(b, N);
    init_matrix(M, N);

    printf("\nVector c:\n");
    print_vector(c, N);

    printf("Vector b:\n");
    print_vector(b, N);

    printf("\nMatriz M:\n");//as
    print_matrix(M, N);
    memcpy(res1, c, sizeof(float)*N);
    memcpy(res2, c, sizeof(float)*N);
    memcpy(res3, c, sizeof(float)*N);


    simple_dgmv(N, res1,M,b);
    printf("\nVector c (simple_dgmv):\n");
    print_vector(res1, N);

    unroll_dgmv(N, res2,M,b);
    printf("\nVector c (unroll_dgmv):\n");
    print_vector(res2, N);

    avx512_dgmv(N, res3,M,b);
    printf("\nVector c (avx512_dgmv):\n");
    print_vector(res3, N);
}