#include "nn/mnist.h"
#include <fstream>
#include <stdexcept>

namespace nn {

    // 大端序转小端序（MNIST 文件是大端存储的）
    static uint32_t read_uint32(std::ifstream& f) {
        uint32_t val = 0;
        unsigned char bytes[4];
        f.read(reinterpret_cast<char*>(bytes), 4);
        val = (bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | bytes[3];
        return val;
    }

    Tensor load_mnist_images(const std::string& filepath) {
        std::ifstream f(filepath, std::ios::binary);
        if (!f.is_open()) {
            throw std::runtime_error("Cannot open file: " + filepath);
        }

        uint32_t magic = read_uint32(f);
        uint32_t num_images = read_uint32(f);
        uint32_t rows = read_uint32(f);
        uint32_t cols = read_uint32(f);

        if (magic != 2051) {
            throw std::runtime_error("Invalid image file magic number");
        }

        size_t image_size = rows * cols;  // 28*28 = 784
        std::vector<float> data(num_images * image_size);

        for (size_t i = 0; i < num_images * image_size; i++) {
            unsigned char pixel;
            f.read(reinterpret_cast<char*>(&pixel), 1);
            data[i] = static_cast<float>(pixel) / 255.0f;  // 归一化到 0~1
        }

        return Tensor({ num_images, image_size }, data);
    }

    std::vector<int> load_mnist_labels(const std::string& filepath) {
        std::ifstream f(filepath, std::ios::binary);
        if (!f.is_open()) {
            throw std::runtime_error("Cannot open file: " + filepath);
        }

        uint32_t magic = read_uint32(f);
        uint32_t num_labels = read_uint32(f);

        if (magic != 2049) {
            throw std::runtime_error("Invalid label file magic number");
        }

        std::vector<int> labels(num_labels);
        for (size_t i = 0; i < num_labels; i++) {
            unsigned char label;
            f.read(reinterpret_cast<char*>(&label), 1);
            labels[i] = static_cast<int>(label);
        }

        return labels;
    }
}