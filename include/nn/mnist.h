#pragma once
#include "tensor.h"
#include <string>
#include <vector>

namespace nn {
    // 读取图片，返回 shape (num_images, 784)，像素值归一化到 0~1
    Tensor load_mnist_images(const std::string& filepath);

    // 读取标签，返回 int 数组
    std::vector<int> load_mnist_labels(const std::string& filepath);
}