#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N 16

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
// only works properly with N multiple of 16
static void avx512_dgmv( size_t n, float c[n], const float M[n][n], const float b[n]) {
    for (int i = 0; i < n; ++i) {
        __m512 sum = _mm512_setzero_ps();
        for (int j = 0; j < n; j += 16) {

            __m512 M_row = _mm512_loadu_ps(&M[i][j]);
            __m512 b_vec = _mm512_loadu_ps(&b[j]);

            sum = _mm512_fmadd_ps(M_row, b_vec, sum);
        }

        c[i] += _mm512_reduce_add_ps(sum);
    }
}

// dgemm
// realiza la operación matricial C = C + A*B, donde A, B y C son tres
// matrices de elementos de tipo double de dimensiones:
// A de dim1 x dim2
// B de dim2 x dim3
// C de dim1 x dim3
void simple_dgemm( int dim1, int dim2, int dim3, float *A, float *B, float *C) {
    for (int i=0; i<dim1; i++){
        for (int j=0; j<dim2; j++){
            for (int k=0; k<dim3; k++){
                //C[i][k] += C[i][k] + A[i][j]*B[j][k];
                *(C+i*dim3+k) += *(A+i*dim2+j) * *(B+j*dim3+k);
            }
        }
    }
}

void avx512_dgemm(int dim1, int dim2, int dim3, float *A, float *B, float *C) {
    for (int i = 0; i < dim1; i++) {
        for (int j = 0; j < dim2; j++) {
            __m512 avec = _mm512_set1_ps(*(A + i * dim2 + j));
            for (int k = 0; k < dim3; k += 16) {

                __m512 bvec = _mm512_loadu_ps(B + j * dim3 + k);

                __m512 cvec = _mm512_loadu_ps(C + i * dim3 + k);

                cvec = _mm512_fmadd_ps(avec, bvec, cvec);

                _mm512_storeu_ps(C + i * dim3 + k, cvec);
            }
        }
    }
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

    printf("------------------------------------------------------------------\n\n\n");
    // dgemm
// realiza la operación matricial C = C + A*B, donde A, B y C son tres
// matrices de elementos de tipo double de dimensiones:
// A de dim1 x dim2
// B de dim2 x dim3
// C de dim1 x dim3

    int dim1,dim2,dim3;
    dim1 = N;
    dim2 = N;
    dim3 = N;

    float A[dim1][dim2];
    float B[dim2][dim3];
    float C[dim1][dim3];

    float res1_1[dim1][dim3];
    float res2_2[dim1][dim3];


    init_matrix(A, N);
    printf("\nMatriz A:\n");//as
    print_matrix(A, N);

    init_matrix(B, N);
    printf("\nMatriz B:\n");//as
    print_matrix(B, N);

    init_matrix(C, N);
    printf("\nMatriz C:\n");//as
    print_matrix(C, N);
    memcpy(res1_1, C, sizeof(float)*dim1*dim3);
    memcpy(res2_2, C, sizeof(float)*dim1*dim3);


    simple_dgemm(dim1,dim2,dim3, A,B,res1_1);
    printf("\nMatriz C (simple_dgemm):\n");
    print_matrix(res1_1, N);

    avx512_dgemm(dim1,dim2,dim3, A,B,res2_2);
    printf("\nMatriz C (avx512_dgemm):\n");
    print_matrix(res2_2, N);
}