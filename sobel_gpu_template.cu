#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <chrono>

using namespace std;

// ------------------------------------
// Sobel Mask (CPU 전역 변수)
// ------------------------------------
int sobelMaskX[] = { -1, 0, 1,
                     -2, 0, 2,
                     -1, 0, 1 };

int sobelMaskY[] = { 1, 2, 1,
                      0, 0, 0,
                     -1,-2,-1 };

// ------------------------------------
// Sobel Mask (GPU 상수 메모리)
// ------------------------------------
__constant__ int d_sobelMaskX[9];
__constant__ int d_sobelMaskY[9];

// ------------------------------------
// 이미지 구조체
// ------------------------------------
struct PPMImage {
    int width, height, maxval;
    unsigned char* data;
};

// ------------------------------------
// CUDA 에러 체크
// ------------------------------------
void checkCuda(cudaError_t err, const char* msg)
{
    if (err != cudaSuccess) {
        cerr << msg << " : " << cudaGetErrorString(err) << endl;
        exit(1);
    }
}

// ------------------------------------
// PPM 읽기 (P6 전용)
// ------------------------------------
PPMImage readPPM(const string& filename)
{
    ifstream fin(filename, ios::binary);
    if (!fin) {
        cerr << "파일을 열 수 없습니다.\n";
        exit(1);
    }

    string format;
    fin >> format;

    if (format != "P6") {
        cerr << "P6 형식만 지원합니다.\n";
        exit(1);
    }

    fin >> ws;
    while (fin.peek() == '#') {
        fin.ignore(10000, '\n');
        fin >> ws;
    }

    PPMImage img;
    fin >> img.width >> img.height >> img.maxval;
    fin.get();

    img.data = new unsigned char[img.width * img.height * 3];
    fin.read(reinterpret_cast<char*>(img.data), img.width * img.height * 3);

    fin.close();
    return img;
}

// ------------------------------------
// RGB -> Gray
// ------------------------------------
unsigned char* rgbToGray(const PPMImage& img)
{
    unsigned char* gray = new unsigned char[img.width * img.height];

    for (int i = 0; i < img.width * img.height; i++)
    {
        int r = img.data[i * 3 + 0];
        int g = img.data[i * 3 + 1];
        int b = img.data[i * 3 + 2];

        gray[i] = (unsigned char)(0.299 * r + 0.587 * g + 0.114 * b + 0.5);
    }

    return gray;
}

// ------------------------------------
// GPU Sobel Kernel
// ------------------------------------
__global__ void sobelGPU(const unsigned char* gray, int W, int H, unsigned char* edge)
{  
}

// ------------------------------------
// main
// ------------------------------------
int main()
{
    string infile = "sample1.ppm";
    string outfile = "output1_gpu.ppm";

    // 1. 이미지 읽기
    PPMImage src = readPPM(infile);

    // 2. Gray 변환
    unsigned char* gray = rgbToGray(src);
    int imgSizeGray = src.width * src.height;

    // 3. GPU 메모리 준비 및 복사
    unsigned char* d_gray = nullptr;
    unsigned char* d_edge = nullptr;
    unsigned char* edge = new unsigned char[imgSizeGray];    
    cudaMemcpyToSymbol(d_sobelMaskX, sobelMaskX, 9 * sizeof(int));  // 상수 메모리로 복사
    cudaMemcpyToSymbol(d_sobelMaskY, sobelMaskY, 9 * sizeof(int));  // 상수 메모리로 복사
    cudaMalloc((void**)&d_gray, imgSizeGray * sizeof(unsigned char));
    cudaMalloc((void**)&d_edge, imgSizeGray * sizeof(unsigned char));
    cudaMemcpy(d_gray, gray, imgSizeGray * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // 4. 커널 실행
    dim3 blockSize(16, 16);
    dim3 gridSize((src.width - 1) / blockSize.x + 1, (src.height - 1) / blockSize.y + 1);

    auto st = chrono::high_resolution_clock::now();
    sobelGPU << <gridSize, blockSize >> > (d_gray, src.width, src.height, d_edge);
    cudaDeviceSynchronize();
    auto ed = chrono::high_resolution_clock::now();

    double ms = chrono::duration<double, milli>(ed - st).count();
    cout << "GPU 시간: " << ms << " ms\n";
       
    // 5. 결과 복사
    cudaMemcpy(edge, d_edge, imgSizeGray * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // 6. Gray -> RGB
    unsigned char* outRGB = new unsigned char[src.width * src.height * 3];
    for (int i = 0; i < imgSizeGray; i++)
    {
        outRGB[i * 3 + 0] = edge[i];
        outRGB[i * 3 + 1] = edge[i];
        outRGB[i * 3 + 2] = edge[i];
    }

    // 7. 저장
    ofstream ofs(outfile, ios::binary);
    ofs << "P6\n" << src.width << " " << src.height << "\n255\n";
    ofs.write(reinterpret_cast<const char*>(outRGB), src.width * src.height * 3);
    ofs.close();
    cout << outfile << " 저장 완료\n";

    // 8. 메모리 해제
    delete[] src.data;
    delete[] gray;
    delete[] edge;
    delete[] outRGB;

    cudaFree(d_gray);
    cudaFree(d_edge);
    return 0;
}