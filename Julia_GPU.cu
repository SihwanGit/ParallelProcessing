#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

__device__ inline unsigned char clampToByte(float x)
{
    return (x < 0) ? 0 : (x > 255 ? 255 : (unsigned char)x);
}

__global__ void juliaKernel(
    unsigned char* image, 
    int width, int height, 
    int maxIter, 
    float xmin, float xmax, float ymin, float ymax, 
    float cx, float cy, float R)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= width || j >= height) return;

    float x = xmin + (xmax - xmin) * i / (width - 1);
    float y = ymax - (ymax - ymin) * j / (height - 1);

    float zx = x, zy = y;
    float R2 = R * R;

    int iter = 0;
    while ((zx * zx + zy * zy) <= R2 && iter < maxIter)
    {
        float new_zx = zx * zx - zy * zy + cx;
        float new_zy = 2.0f * zx * zy + cy;
        zx = new_zx;
        zy = new_zy;
        ++iter;
    }

    int idx = 3 * (j * width + i);
    if (iter == maxIter)
    {
        image[idx + 0] = image[idx + 1] = image[idx + 2] = 0;
    }
    else
    {
        float t = (float)iter / maxIter;
        image[idx + 0] = (unsigned char)(9 * (1 - t) * t * t * t * 255);
        image[idx + 1] = (unsigned char)(15 * (1 - t) * (1 - t) * t * t * 255);
        image[idx + 2] = (unsigned char)(8.5 * (1 - t) * (1 - t) * (1 - t) * t * 255);
    }
}

int main()
{
    const int width = 1920;
    const int height = 1080;
    const int maxIter = 300;

    //const float cx = -0.8f, cy = 0.156f;
    //const float cx = 0.285f, cy = 0.0f;
    //const float cx = 0.0f, cy = 0.0f;
    const float cx = -0.70176, cy = -0.3842;

    const float xmin = -1.8f, xmax = 1.8f;
    const float ymin = -1.2f, ymax = 1.2f;
    const float R = 2.0f;

    size_t imageSize = width * height * 3;

    std::vector<unsigned char> h_image(imageSize);
    unsigned char* d_image;
    cudaMalloc(&d_image, imageSize);

    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);
    auto total_start = std::chrono::high_resolution_clock::now();
    {
        juliaKernel << <grid, block >> > (d_image, width, height, maxIter, xmin, xmax, ymin, ymax, cx, cy, R);
        cudaDeviceSynchronize();
        cudaMemcpy(h_image.data(), d_image, imageSize, cudaMemcpyDeviceToHost);
    }   
    auto total_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> total_ms = total_end - total_start;    
    std::cout << "Total time: " << total_ms.count() << " ms\n";

    // 저장
    std::ofstream ofs("julia_cuda.ppm", std::ios::binary);
    ofs << "P6\n" << width << " " << height << "\n255\n";
    ofs.write((char*)h_image.data(), h_image.size());
    ofs.close();    

    // cleanup
    cudaFree(d_image);
    return 0;
}