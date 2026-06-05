#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <math.h>
#include <iostream>

#define Size (1024 * 1024 * 512)       // 2GB
//#define Size (1024 * 1024 * 256)       // 1GB
//#define Size (1024 * 1024 * 128)
#define NumThread 512

__global__ void parallel_sum1(float* dev_in, float* dev_out, int size);
__global__ void parallel_sum2(float* dev_in, float* dev_out, int size);

double gpu_reduce_sum(float* dev_src, int size, bool less_branch);

int main()
{
    // CPU 메모리 할당 및 초기화
    float* M = new float[Size];
    for (int i = 0; i < Size; i++)
        M[i] = rand() / (float)RAND_MAX;
    printf("초기화 완료...\n\n");

    // GPU 메모리 할당 및 복사
    cudaSetDevice(0);

    float* dev_M1, * dev_M2;
    cudaMalloc((void**)&dev_M1, Size * sizeof(float));
    cudaMalloc((void**)&dev_M2, Size * sizeof(float));
    cudaMemcpy(dev_M1, M, Size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_M2, M, Size * sizeof(float), cudaMemcpyHostToDevice);

    // CPU 순차 합
    auto st = std::chrono::high_resolution_clock::now();
    double cpu_sum = 0.0;
    for (int i = 0; i < Size; ++i)
        cpu_sum += M[i];
    auto ed = std::chrono::high_resolution_clock::now();    
    double cpu_ms = std::chrono::duration<double, std::milli>(ed - st).count();

    printf("CPU 경과시간 = %lld ms\n", (long long)cpu_ms);    
    printf("순차 합 = %.3lf\n\n", cpu_sum);
      
    // GPU (더 분기)
    st = std::chrono::high_resolution_clock::now();
    double gpu1_sum = gpu_reduce_sum(dev_M1, Size, false);
    ed = std::chrono::high_resolution_clock::now();
    double gpu1_ms = std::chrono::duration<double, std::milli>(ed - st).count();

    printf("GPU 경과시간(더 분기) = %lld ms\n", (long long)gpu1_ms);
    printf("병렬 합(더 분기) = %.3lf\n\n", gpu1_sum);

    // GPU (덜 분기)
    st = std::chrono::high_resolution_clock::now();
    double gpu2_sum = gpu_reduce_sum(dev_M2, Size, true);
    ed = std::chrono::high_resolution_clock::now();
    double gpu2_ms = std::chrono::duration<double, std::milli>(ed - st).count();

    printf("GPU 경과시간(덜 분기) = %lld ms\n", (long long)gpu2_ms);
    printf("병렬 합(덜 분기) = %.3lf\n\n", gpu2_sum);

    // 메모리 해제
    cudaFree(dev_M1);
    cudaFree(dev_M2);
    delete[] M;
    cudaDeviceReset();

    return 0;
}

__global__ void parallel_sum1(
    float* dev_in, float* dev_out, int size)
{
    __shared__ float partialSum[NumThread];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // 현재 블록의 데이터를 공유 메모리로 복사
    partialSum[tid] = (i < size) ? dev_in[i] : 0.0f;
    __syncthreads();

    // 분기가 많은 병렬 합 수행
    for (unsigned int s = 1; s < blockDim.x; s *= 2)
    {
        if ((tid % (2 * s)) == 0)
            partialSum[tid] += partialSum[tid + s];
        __syncthreads();
    }

    // 현재 블록의 결과를 전역 메모리에 기록
    if (tid == 0)
        dev_out[blockIdx.x] = partialSum[0];
}

__global__ void parallel_sum2(float* dev_in, float* dev_out, int size)
{
    __shared__ float partialSum[NumThread];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    partialSum[tid] = (i < size) ? dev_in[i] : 0.0f;
    __syncthreads();

    // do reduction in shared memory (덜 분기)
    for (int s = blockDim.x / 2; s > 0; s /= 2)
    {
        if (tid < s)
            partialSum[tid] += partialSum[tid + s];
        __syncthreads();
    }

    // write result for this block to global memory
    if (tid == 0)
        dev_out[blockIdx.x] = partialSum[0];
}

double gpu_reduce_sum(float* dev_src, int size, bool less_branch)
{
    float* dev_in = dev_src;
    float* dev_out = nullptr;

    int curSize = size;

    while (true)
    {
        int gridSize = (curSize - 1) / NumThread + 1;
        cudaMalloc((void**)&dev_out, gridSize * sizeof(float));

        dim3 dimGrid(gridSize, 1);
        dim3 dimBlock(NumThread, 1, 1);

        if (less_branch)
            parallel_sum2 << <dimGrid, dimBlock >> > (dev_in, dev_out, curSize);
        else
            parallel_sum1 << <dimGrid, dimBlock >> > (dev_in, dev_out, curSize);
        cudaDeviceSynchronize();

        if (dev_in != dev_src)
            cudaFree(dev_in);

        if (gridSize == 1)
            break;

        dev_in = dev_out;
        curSize = gridSize;
    }

    float result = 0.0f;
    cudaMemcpy(&result, dev_out, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(dev_out);

    return (double)result;
}
