#include "nn/tensor.h"
#include "nn/layer.h"
#include "nn/loss.h"
#include "nn/optimizer.h"
#include "nn/mnist.h"
#include <iostream>

int main() {
    // ========== 加载数据 ==========
    std::cout << "Loading data...\n";
    nn::Tensor train_images = nn::load_mnist_images("C:/Users/86157/Desktop/nn_engine/data/train-images.idx3-ubyte");
    std::vector<int> train_labels = nn::load_mnist_labels("C:/Users/86157/Desktop/nn_engine/data/train-labels.idx1-ubyte");
    nn::Tensor train_targets = nn::Tensor::one_hot(train_labels, 10);

    nn::Tensor test_images = nn::load_mnist_images("C:/Users/86157/Desktop/nn_engine/data/t10k-images.idx3-ubyte");
    std::vector<int> test_labels = nn::load_mnist_labels("C:/Users/86157/Desktop/nn_engine/data/t10k-labels.idx1-ubyte");

    std::cout << "Train: " << train_images.shape()[0] << " images\n";
    std::cout << "Test:  " << test_images.shape()[0] << " images\n";

    // ========== 网络结构 ==========
    nn::Sequential model({
        std::make_shared<nn::Linear>(784, 128),
        std::make_shared<nn::ReLU>(),
        std::make_shared<nn::Dropout>(0.2f),
        std::make_shared<nn::Linear>(128, 64),
        std::make_shared<nn::ReLU>(),
        std::make_shared<nn::Dropout>(0.2f),
        std::make_shared<nn::Linear>(64, 10),
        std::make_shared<nn::Softmax>()
        });

    nn::CrossEntropyLoss criterion;
    nn::Adam optimizer(model.parameters(), 0.001f);

    // ========== 训练 ==========
    size_t batch_size = 64;
    size_t num_batches = train_images.shape()[0] / batch_size;
    int epochs = 10;

    for (int epoch = 0; epoch < epochs; epoch++) {
        float total_loss = 0.0f;
        model.train();

        for (size_t b = 0; b < num_batches; b++) {
            size_t start = b * batch_size;
            size_t end = start + batch_size;
            nn::Tensor x_batch = train_images.slice(start, end);
            nn::Tensor y_batch = train_targets.slice(start, end);

            optimizer.zero_grad();
            nn::Tensor pred = model.forward(x_batch);
            nn::Tensor loss = criterion.forward(pred, y_batch);
            loss.backward();
            optimizer.step();

            total_loss += loss.data_ptr()[0];

            if (b % 100 == 0) {
                std::cout << "Epoch " << epoch + 1
                    << " Batch " << b << "/" << num_batches
                    << " Loss: " << loss.data_ptr()[0] << "\n";
            }
        }

        std::cout << "Epoch " << epoch + 1
            << " Avg Loss: " << total_loss / num_batches << "\n";

        // ========== 测试准确率 ==========
        model.eval();
        int correct = 0;
        size_t test_batches = test_images.shape()[0] / batch_size;

        for (size_t b = 0; b < test_batches; b++) {
            size_t start = b * batch_size;
            size_t end = start + batch_size;
            nn::Tensor x_batch = test_images.slice(start, end);

            nn::Tensor pred = model.forward(x_batch);
            nn::Tensor pred_labels = pred.argmax(1);

            for (size_t j = 0; j < batch_size; j++) {
                if (static_cast<int>(pred_labels.data_ptr()[j]) == test_labels[start + j]) {
                    correct++;
                }
            }
        }

        float accuracy = static_cast<float>(correct) / (test_batches * batch_size) * 100.0f;
        std::cout << "Accuracy: " << accuracy << "%\n\n";
    }

    return 0;
}