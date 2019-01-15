#include <math.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

// ld : leading dimension
// _GET_ADDR_N : no transpose
// _GET_ADDR_T : transpose
#define _GET_ADDR_N(row, col, ld) (col * ld + row)
#define _GET_ADDR_T(row, col, ld) (row * ld + col)
#define _OUT_OF_RANGE(row, row_range, col, col_range) (row >= row_range || col >= col_range)
#define _GET_BLOCK_NUM(x, block_size) (x + block_size - 1) / block_size

// C = alpha * op(A) * op(B) + beta * C
// handle : not used, just keep same with cublasSgemm
template <int BLOCK_SIZE> __global__ void
matrixMul_2(cublasHandle_t handle,
            cublasOperation_t transa, cublasOperation_t transb,
            int m, int n, int k,
            const float alpha,   // use (float*) seems wrong, why ?
            const float *A, int lda,
            const float *B, int ldb,
            const float beta,    // use (float*) seems wrong, why ?
            float *C, int ldc)
{
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Steps each thread, k is col of op(A) or row of op(B)
    int steps = int((k + BLOCK_SIZE - 1) / (BLOCK_SIZE));

    // Csub is used to shtore the element of the block sub-matrix
    // that is computed by the thread
    float Csub = 0.0;

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for(int step = 0; step < steps; ++step)
    {
        // Declaration of the shared memory array As used to
        // store the sub-matrix of A and B
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load the matrices from device memory
        // to shared memory, each thread loads
        // one element of each matrix
        int a_x = BLOCK_SIZE * step + tx;
        int a_y = BLOCK_SIZE * by   + ty;
        int a_addr = (transa == CUBLAS_OP_T) ? _GET_ADDR_T(a_y, a_x, lda)
                                             : _GET_ADDR_N(a_y, a_x, lda);
        As[ty][tx] = _OUT_OF_RANGE(a_y, m, a_x, k) ? 0.0
                                                   : A[a_addr];

        int b_x = BLOCK_SIZE * bx   + tx;
        int b_y = BLOCK_SIZE * step + ty;
        int b_addr = (transb == CUBLAS_OP_T) ? _GET_ADDR_T(b_y, b_x, ldb)
                                             : _GET_ADDR_N(b_y, b_x, ldb);
        Bs[ty][tx] = _OUT_OF_RANGE(b_y, k, b_x, n) ? 0.0
                                                   : B[b_addr];

        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrx
        for(int bs = 0; bs < BLOCK_SIZE; ++bs)
        {
            Csub += As[ty][bs] * Bs[bs][tx];
        }

        // Synchronize to make sure that the preceding
        // computation is done befroe laoding two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    int c_x = bx * BLOCK_SIZE + tx;
    int c_y = by * BLOCK_SIZE + ty;
    int c_addr = _GET_ADDR_N(c_y, c_x, ldc);
    if(!_OUT_OF_RANGE(c_y, m, c_x, n)) {
        //C[c_addr] = (*alpha) * Csub + (*beta) * C[c_addr];
        C[c_addr] = alpha * Csub + beta * C[c_addr];
    }
}

__global__ void
matrixMul_(cublasHandle_t handle,
           cublasOperation_t transA, cublasOperation_t transB,
           int M, int N, int K,
           const float alpha,
           const float *A, int lda,
           const float *B, int ldb,
           const float beta,
           float *C, int ldc)
{
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    float sum = 0.0;
    for(int i = 0; i < K; i++) {
        int a_addr = (transA == CUBLAS_OP_T) ? _GET_ADDR_T(by, i, lda) : _GET_ADDR_N(by, i, lda);
        int b_addr = (transB == CUBLAS_OP_T) ? _GET_ADDR_T(i, bx, ldb) : _GET_ADDR_N(i, bx, ldb);
        sum += A[a_addr] * B[b_addr];
    }

    int c_addr = _GET_ADDR_N(by, bx, ldc);
    C[c_addr] = alpha * sum + beta * C[c_addr];
}

void test()
{
    const int M = 128;
    const int N = 128;
    const int K = 128;
    const int block_size = 16;
    const int max_matrix = 128;

    // initilize
    float h_a[M][K], h_b[K][N], h_c[M][N];
    float h_rst_1[M][N], h_rst_2[M][N];
    for(int i = 0; i < M; i++) {
        for(int j = 0; j < K; j++) {
            h_a[i][j] = 4.3 * i + 1.9 * j;
        }
    }
    for(int i = 0; i < K; i++) {
        for(int j = 0; j < N; j++) {
            h_b[i][j] = 2.0 * i + 5.3*j;
        }
    }
    for(int i = 0; i < M; i++) {
        for(int j = 0; j < N; j++) {
            h_c[i][j] = 3.2 * i + 4.5 * j;
        }
    }

    // allocage device memory
    float *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, M*K*sizeof(float));
    cudaMalloc((void**)&d_b, K*N*sizeof(float));
    cudaMalloc((void**)&d_c, M*N*sizeof(float));

    cudaMemcpy(d_a, h_a, M*K*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, K*N*sizeof(float), cudaMemcpyHostToDevice);

    int total_diff_num = 0;
    float alpha = 2.3, beta = 2.3, max_diff = 0.0;

    // cublas
    cublasHandle_t handle;
    cublasStatus_t status = cublasCreate(&handle);
    cublasOperation_t op[2] = {CUBLAS_OP_N, CUBLAS_OP_T};

    for(int ta = 0; ta < 2; ta++) {
        for(int tb = 0; tb < 2; tb++) {
            for(int mm = 1; mm < max_matrix; mm += 1) {
                for(int nn = 1; nn < max_matrix; nn += 1) {
                    for(int kk = 1; kk < max_matrix; kk += 1) {

                        int lda = (op[ta] == CUBLAS_OP_T) ? kk : mm;
                        int ldb = (op[tb] == CUBLAS_OP_T) ? nn : kk;

                        dim3 threads(block_size, block_size);
                        dim3 grid(int(_GET_BLOCK_NUM(nn, block_size)), int(_GET_BLOCK_NUM(mm, block_size)));

                        // cublassgemm
                        memset((void*)h_rst_1, 0.0, M * N * sizeof(float));
                        cudaMemcpy(d_c, h_c, M * N * sizeof(float), cudaMemcpyHostToDevice);
                        cublasSgemm(handle, op[ta], op[tb],
                                    mm, nn, kk,
                                    &alpha,
                                    d_a, lda,
                                    d_b, ldb,
                                    &beta,
                                    d_c, mm);
                        cudaMemcpy(h_rst_1, d_c, mm * nn * sizeof(float), cudaMemcpyDeviceToHost);

                        const bool use_opt = true;
                        if(use_opt) {
                            // matrixMul_2
                            memset((void*)h_rst_2, 0.0, M * N * sizeof(float));
                            cudaMemcpy(d_c, h_c, M * N * sizeof(float), cudaMemcpyHostToDevice);
                            matrixMul_2<block_size><<<grid, threads>>>(handle, op[ta], op[tb],
        	                				                 mm, nn, kk,
                                                             alpha,
                                                             d_a, lda,
                                                             d_b, ldb,
                                                             beta,
                                                             d_c, mm);
                            cudaMemcpy(h_rst_2, d_c, mm * nn * sizeof(float), cudaMemcpyDeviceToHost);
                        } else {
                            // matrixMul_, no opt
                            dim3 grids2(nn, mm, 1);
                            dim3 threads2(1, 1, 1);
                            memset((void*)h_rst_2, 0.0, M * N * sizeof(float));
                            cudaMemcpy(d_c, h_c, M * N * sizeof(float), cudaMemcpyHostToDevice);
                            matrixMul_<<<grids2, threads2>>>(handle, op[ta], op[tb],
                                                             mm, nn, kk,
                                                             alpha,
                                                             d_a, lda,
                                                             d_b, ldb,
                                                             beta,
                                                             d_c, mm);
                            cudaMemcpy(h_rst_2, d_c, mm * nn * sizeof(float), cudaMemcpyDeviceToHost);
                        }

                        // compare
                        for(int rst_y = 0; rst_y < M; rst_y++) {
                            for(int rst_x = 0; rst_x < N; rst_x++) {
                                max_diff = fmax(max_diff, fabsf(h_rst_1[rst_y][rst_x] - h_rst_2[rst_y][rst_x]));

                                if(h_rst_1[rst_y][rst_x] - h_rst_2[rst_y][rst_x] >  3.0 ||
                                   h_rst_1[rst_y][rst_x] - h_rst_2[rst_y][rst_x] < -3.0) {
                                    total_diff_num ++;
                                    printf("m = %d, n = %d, k = %d, row = %d, col = %d, rst_1 = %f, rst_2 = %f\n",
                                                mm,     nn,     kk, rst_y,    rst_x, h_rst_1[rst_y][rst_x], h_rst_2[rst_y][rst_x]);
                                }
                            }
                        }

                    }
                }
            }
        }
    }

    if(total_diff_num <= 0) {
        printf("\n\n======PASS======\n\n");
    } else {
        printf("total_diff_num = %d\n", total_diff_num);
    }
    printf("\nmax_diff = %.7f\n\n", max_diff);

    // release
    cublasDestroy(handle);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

int main()
{
    int devID = 2;
    cudaError_t error = cudaGetDevice(&devID);

    test();

    return 0;
}
