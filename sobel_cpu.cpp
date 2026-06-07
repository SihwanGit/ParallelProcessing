#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <chrono>

using namespace std;

// ------------------------------------
// Sobel Mask (전역 변수)
// ------------------------------------
int sobelMaskX[] = { -1, 0, 1,
                     -2, 0, 2,
                     -1, 0, 1 };

int sobelMaskY[] = { 1, 2, 1,
                      0, 0, 0,
                     -1,-2,-1 };

// ------------------------------------
// 이미지 구조체
// ------------------------------------
struct PPMImage {
    int width, height, maxval;
    unsigned char* data;
};

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
    fin.read(reinterpret_cast<char*>(img.data),
        img.width * img.height * 3);

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
        gray[i] = (unsigned char) (0.299 * r + 0.587 * g + 0.114 * b + 0.5);
    }
    return gray;
}

// ------------------------------------
// Sobel Edge Detection
// ------------------------------------
void sobel(const unsigned char* gray, int W, int H, unsigned char* edge)
{
    for (int i = 0; i < W * H; i++)
        edge[i] = 0;

    for (int y = 1; y < H - 1; y++)
    {
        for (int x = 1; x < W - 1; x++)
        {
            int gx = 0;
            int gy = 0;

            for (int k = 0; k < 9; k++)
            {
                int dx = k % 3 - 1;   // -1, 0, 1, -1, 0, 1, -1, 0, 1
                int dy = k / 3 - 1;   // -1, -1, -1, 0, 0, 0, 1, 1, 1

                int pixel = gray[(y + dy) * W + (x + dx)];
                gx += pixel * sobelMaskX[k];
                gy += pixel * sobelMaskY[k];
            }

            int mag = (int)(sqrt((double)(gx * gx + gy * gy)));
            if (mag > 255) mag = 255;

            edge[y * W + x] = (unsigned char)mag;
        }
    }
}

// ------------------------------------
// main
// ------------------------------------
int main()
{
    string infile = "sample2.ppm";
    string outfile = "output2.ppm";

    // 1. 이미지 읽기
    PPMImage src = readPPM(infile);

    // 2. Gray 변환
    unsigned char* gray = rgbToGray(src);

    // 3. Sobel
    unsigned char* edge = new unsigned char[src.width * src.height];

    auto st = chrono::high_resolution_clock::now();
    sobel(gray, src.width, src.height, edge);
    auto ed = chrono::high_resolution_clock::now();

    double ms = chrono::duration<double, milli>(ed - st).count();
    cout << "CPU 시간: " << ms << " ms\n";

    // 4. Gray -> RGB
    unsigned char* outRGB = new unsigned char[src.width * src.height * 3];

    for (int i = 0; i < src.width * src.height; i++)
    {
        outRGB[i * 3 + 0] = edge[i];
        outRGB[i * 3 + 1] = edge[i];
        outRGB[i * 3 + 2] = edge[i];
    }

    // 5. 저장
    ofstream ofs(outfile, ios::binary);
    ofs << "P6\n" << src.width << " " << src.height << "\n" << 255 << "\n";
    ofs.write(reinterpret_cast<const char*>(outRGB), src.width * src.height * 3);
    ofs.close();
    cout << outfile << " 저장 완료\n";

    // 6. 메모리 해제
    delete[] src.data;
    delete[] gray;
    delete[] edge;
    delete[] outRGB;

    return 0;
}