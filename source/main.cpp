#include "nn/tensor.h"
#include "nn/layer.h"
#include "nn/loss.h"
#include "nn/optimizer.h"
#include <iostream>
#include <cmath>

int main() {
    // 训练数据：y = x^2，x 从 -2 到 2，共 20 个点
    std::vector<float> x_data, y_data;
    for (int i = 0; i < 20; i++) {
        float x = -2.0f + i * 0.2f;
        x_data.push_back(x);
        y_data.push_back(x * x);
    }
    nn::Tensor x({ 20, 1 }, x_data);
    nn::Tensor y({ 20, 1 }, y_data);

    // 网络：Linear(1,16) → ReLU → Linear(16,8) → ReLU → Linear(8,1)
    nn::Sequential model({
        std::make_shared<nn::Linear>(1, 16),
        std::make_shared<nn::ReLU>(),
        std::make_shared<nn::Linear>(16, 8),
        std::make_shared<nn::ReLU>(),
        std::make_shared<nn::Linear>(8, 1)
        });

    nn::MSELoss criterion;
    nn::SGD optimizer(model.parameters(), 0.01f);

    // 训练 2000 轮
    for (int epoch = 0; epoch < 6000; epoch++) {
        optimizer.zero_grad();
        nn::Tensor pred = model.forward(x);
        nn::Tensor loss = criterion.forward(pred, y);
        loss.backward();
        optimizer.step();

        if (epoch % 200 == 0) {
            std::cout << "Epoch " << epoch
                << "  Loss: " << loss.data_ptr()[0] << "\n";
        }
    }

    // 测试几个点
    std::cout << "\nTest:\n";
    for (float val : {-1.5f, 0.0f, 1.0f, 1.5f}) {
        nn::Tensor test({ 1, 1 }, { val });
        nn::Tensor out = model.forward(test);
        std::cout << "x=" << val
            << "  pred=" << out.data_ptr()[0]
            << "  true=" << val * val << "\n";
    }
}