#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <iomanip>

#define TILE_WIDTH 16

// ------------------------------------------------------------
// CUDA 에러 체크 매크로
// ------------------------------------------------------------
#define CUDA_CHECK(call)                                                   \
do {                                                                       \
    cudaError_t err = (call);                                              \
    if (err != cudaSuccess) {                                              \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err)             \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl;   \
        std::exit(EXIT_FAILURE);                                           \
    }                                                                      \
} while (0)

// ------------------------------------------------------------
// Naive 전역 메모리 버전
// ------------------------------------------------------------
__global__ void MultMatGPU_Naive(float* P, const float* M, const float* N, int width)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < width && j < width)
    {
        float sum = 0.0f;
        for (int k = 0; k < width; ++k)
        {
            float a = M[i * width + k];
            float b = N[k * width + j];
            sum += a * b;
        }
        P[i * width + j] = sum;
    }
}

// ------------------------------------------------------------
// Shared Memory Tiled 버전
// TILE_WIDTH x TILE_WIDTH 블록 가정
// ------------------------------------------------------------
__global__ void MultMatGPU_Tiled(float* P, const float* M, const float* N, int width)
{
    __shared__ float sM[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sN[TILE_WIDTH][TILE_WIDTH];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * TILE_WIDTH + ty;
    int col = blockIdx.x * TILE_WIDTH + tx;

    float sum = 0.0f;

    int numTiles = (width + TILE_WIDTH - 1) / TILE_WIDTH;

    for (int m = 0; m < numTiles; ++m)
    {
        int tiledColM = m * TILE_WIDTH + tx;
        int tiledRowN = m * TILE_WIDTH + ty;

        if (row < width && tiledColM < width)
            sM[ty][tx] = M[row * width + tiledColM];
        else
            sM[ty][tx] = 0.0f;

        if (tiledRowN < width && col < width)
            sN[ty][tx] = N[tiledRowN * width + col];
        else
            sN[ty][tx] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k)
            sum += sM[ty][k] * sN[k][tx];

        __syncthreads();
    }

    if (row < width && col < width)
        P[row * width + col] = sum;
}

// 결과 비교
bool CompareMat(const std::vector<float>& A, const std::vector<float>& B, float eps = 1e-3f)
{
    if (A.size() != B.size()) return false;

    for (size_t i = 0; i < A.size(); ++i)
    {
        if (std::fabs(A[i] - B[i]) > eps)
        {
            std::cout << "Mismatch at index " << i
                << " : A = " << A[i]
                << ", B = " << B[i] << std::endl;
            return false;
        }
    }
    return true;
}

int main()
{
    // 설정
    const int width = 2048;
    const size_t bytes = sizeof(float) * width * width;
    std::cout << "Matrix size: " << width << " x " << width << std::endl;

    // Host 메모리 할당 및 초기화
    std::vector<float> M(width * width);
    std::vector<float> N(width * width);
    std::vector<float> P_naive(width * width, 0.0f);
    std::vector<float> P_tiled(width * width, 0.0f);

    std::srand(0);
    for (int i = 0; i < width * width; ++i) {
        M[i] = static_cast<float>(std::rand() % 3 - 1); // {-1, 0, 1}
        N[i] = static_cast<float>(std::rand() % 3 - 1);
    }

    // GPU 선택    
    CUDA_CHECK(cudaSetDevice(0));

    // Device 메모리 할당    
    float* devM = nullptr;
    float* devN = nullptr;
    float* devP_naive = nullptr;
    float* devP_tiled = nullptr;

    CUDA_CHECK(cudaMalloc((void**)&devM, bytes));
    CUDA_CHECK(cudaMalloc((void**)&devN, bytes));
    CUDA_CHECK(cudaMalloc((void**)&devP_naive, bytes));
    CUDA_CHECK(cudaMalloc((void**)&devP_tiled, bytes));

    // Host -> Device 복사    
    CUDA_CHECK(cudaMemcpy(devM, M.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(devN, N.data(), bytes, cudaMemcpyHostToDevice));

    // 1. Naive 버전 실행 및 시간 측정    
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 dimGrid((width + TILE_WIDTH - 1) / TILE_WIDTH, (width + TILE_WIDTH - 1) / TILE_WIDTH, 1);

    CUDA_CHECK(cudaMemset(devP_naive, 0, bytes));

    auto startNaive = std::chrono::high_resolution_clock::now();
    MultMatGPU_Naive << <dimGrid, dimBlock >> > (devP_naive, devM, devN, width);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    auto endNaive = std::chrono::high_resolution_clock::now();
    double naiveMs = std::chrono::duration<double, std::milli>(endNaive - startNaive).count();

    CUDA_CHECK(cudaMemcpy(P_naive.data(), devP_naive, bytes, cudaMemcpyDeviceToHost));

    // 2. Tiled 버전 실행 및 시간 측정        
    CUDA_CHECK(cudaMemset(devP_tiled, 0, bytes));
    auto startTiled = std::chrono::high_resolution_clock::now();
    MultMatGPU_Tiled << <dimGrid, dimBlock >> > (devP_tiled, devM, devN, width);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    auto endTiled = std::chrono::high_resolution_clock::now();
    double tiledMs = std::chrono::duration<double, std::milli>(endTiled - startTiled).count();

    CUDA_CHECK(cudaMemcpy(P_tiled.data(), devP_tiled, bytes, cudaMemcpyDeviceToHost));

    // 결과 검증    
    bool same = CompareMat(P_naive, P_tiled);

    // 출력    
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "\n[Execution Time]\n";
    std::cout << "Naive  kernel : " << naiveMs << " ms\n";
    std::cout << "Tiled  kernel : " << tiledMs << " ms\n";

    if (tiledMs > 0.0)
        std::cout << "Speedup        : " << (naiveMs / tiledMs) << " x\n";

    std::cout << "\n[Verification]\n";
    std::cout << "Result match   : " << (same ? "PASS" : "FAIL") << std::endl;

    // 메모리 해제
    CUDA_CHECK(cudaFree(devP_tiled));
    CUDA_CHECK(cudaFree(devP_naive));
    CUDA_CHECK(cudaFree(devN));
    CUDA_CHECK(cudaFree(devM));

    CUDA_CHECK(cudaDeviceReset());
    return 0;
}