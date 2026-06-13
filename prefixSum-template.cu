
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <chrono>

#define BLOCK_SIZE   256
#define ARRAY_SIZE   65536 // 256 * 256

/*!
 *  \brief  Brent-Kung 알고리즘으로 한 블록의 prefix 합을 계산
 *
 *  \param  X[in]           입력 배열(GPU)
 *  \param  Y[out]          prefix 합을 저장하는 출력 배열(GPU)
 *  \param  blockSum[out]   nullptr이 아니면 각 블록의 합이 저장됨
 *  \param  n[in]           입력 배열의 크기
 */
__global__ void scanBrentKung(const int* X, int* Y, int* blockSum, int n)
{
    // 공유 메모리
    __shared__ int s[BLOCK_SIZE];

    int tid  = threadIdx.x; // 블록내 인덱스
    int gid  = blockIdx.x * BLOCK_SIZE + tid; // 전역 인덱스

    // 전역 메모리를 공유 메모리로 복사(범위 초과 시 0으로 패딩)
    s[tid] = (gid < n) ? X[gid] : 0;
    __syncthreads();

    // Up-sweep
    for (int stride = 1; stride <= BLOCK_SIZE / 2; stride *= 2)
    {
        __syncthreads();
        int idx = (tid + 1) * 2 * stride - 1;
        if (idx < BLOCK_SIZE)
            s[idx] += s[idx - stride];
    }

    // 블록 합 저장
    // up-sweep 후 s[BLOCK_SIZE-1] 에 블록 전체 합이 모임
    if (tid == 0 && blockSum != nullptr)
        blockSum[blockIdx.x] = s[BLOCK_SIZE - 1];

    // Down-sweep
    for (int stride = BLOCK_SIZE / 4; stride >= 1; stride /= 2)
    {
        __syncthreads();
        int idx = (tid + 1) * 2 * stride - 1;
        if (idx + stride < BLOCK_SIZE)
            s[idx + stride] += s[idx];
    }
    __syncthreads();

    // 공유 메모리 -> 전역 메모리
    if (gid < n)
        Y[gid] = s[tid];
}

/*!
 *  \brief  블록의 prefix 합을 전파
 *
 *  \param  Y[out] 		    prefix 합을 저장하는 출력 배열(GPU)
 *  \param  blockSum[in]    블록의 prefix 합을 저장한 배열
 *  \param  n[in]       	입력 배열의 크기
 */
__global__ void propagate(int* Y, const int* blockSum, int n)
{
    if (blockIdx.x == 0) 
        return;  // 블록 0 은 보정 불필요

    // 구현 하세요.
}

/*!
 *  \brief   Brent-Kung 알고리즘으로 prefix 합을 계산한다.
 * 
 *  \param  X[in]   입력 배열의 GPU 메모리 주소 
 *  \param  Y[in]   출력 배열의 GPU 메모리 주소
 *  \param  N[in]   입력 배열의 크기
 */
void prefixSum(const int* X, int* Y, int N)
{
    // 블록 수 계산(256)
    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // 블록의 부분합을 저장할 GPU 배열
    int* blockSums = nullptr;
    cudaMalloc(&blockSums, numBlocks * sizeof(int));

    // 1단계 스캔(blockSums에 각 블록의 합이 저장됨)
    
    // 2단계 스캔(blockSums의 prefix 합을 계산)    
    
    // 3단계 전파(blockSums의 결과를 Y에 전파)   

    // GPU 메모리 해제
    cudaFree(blockSums);
}

int main()
{
    const int N = ARRAY_SIZE;
    const size_t bytes = (size_t)N * sizeof(int);

    // CPU 메모리 할당
    int* h_in  = (int*)malloc(bytes);
    int* h_out = (int*)malloc(bytes);

    // 입력 배열 초기화: 모두 1
    for (int i = 0; i < N; i++) 
        h_in[i] = 1;

    // GPU 메모리 할당 및 복사
    int* d_in  = nullptr;
    int* d_out = nullptr;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    // 워밍업(첫 번째, CUDA 호출은 시간이 좀 더 걸림)
    prefixSum(d_in, d_out, N);
    cudaDeviceSynchronize();

    // 커널 함수 호출 및 시간 측정
    auto t0 = std::chrono::high_resolution_clock::now();
    prefixSum(d_in, d_out, N);
    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("경과 시간  : %.3f ms\n", ms);

    // 결과 검증
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; i++) 
    {
        if (h_out[i] != i + 1) 
        {
            printf("[FAIL] index=%d  got=%d  expected=%d\n", i, h_out[i], i + 1);
            return -1;
        }
    }
    printf("[PASS] 전체 %d 원소 검증 완료\n", N);    

    // 정리
    free(h_in);  
    free(h_out);
    cudaFree(d_in); 
    cudaFree(d_out);
    return 0;
}
