#include <iostream>
#include <fstream>
#include <complex>
#include <vector>
#include <cmath>
#include <chrono>   // 추가

int main()
{
    const int width = 1920;
    const int height = 1080;
    const int maxIter = 300;

    const std::complex<double> c(-0.8, 0.156);

    const double xmin = -1.8;
    const double xmax = 1.8;
    const double ymin = -1.2;
    const double ymax = 1.2;
    const double R = 2.0;

    std::vector<unsigned char> image(width * height * 3);

    // ============================
    // ⏱️ 시간 측정 시작
    // ============================
    auto start = std::chrono::high_resolution_clock::now();

    for (int j = 0; j < height; ++j) {
        for (int i = 0; i < width; ++i) {
            double x = xmin + (xmax - xmin) * i / (width - 1);
            double y = ymax - (ymax - ymin) * j / (height - 1);
            std::complex<double> z(x, y);

            int iter = 0;
            while (std::abs(z) <= R && iter < maxIter) {
                z = z * z + c;
                ++iter;
            }

            int idx = 3 * (j * width + i);

            if (iter == maxIter) {
                image[idx + 0] = image[idx + 1] = image[idx + 2] = 0;
            }
            else {
                double t = (double)iter / maxIter;
                image[idx + 0] = (unsigned char)(9 * (1 - t) * t * t * t * 255);
                image[idx + 1] = (unsigned char)(15 * (1 - t) * (1 - t) * t * t * 255);
                image[idx + 2] = (unsigned char)(8.5 * (1 - t) * (1 - t) * (1 - t) * t * 255);
            }
        }
    }

    // ============================
    // ⏱️ 시간 측정 종료
    // ============================
    auto end = std::chrono::high_resolution_clock::now();    
    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << "Render time: " << elapsed.count() << " ms\n";

    // 이미지 저장
    std::ofstream ofs("julia.ppm", std::ios::binary);
    ofs << "P6\n" << width << " " << height << "\n255\n";
    ofs.write(reinterpret_cast<const char*>(image.data()), image.size());
    ofs.close();

    std::cout << "Saved julia.ppm\n";

    return 0;
}
