#include "nn/tensor.h"

int main() {
    // test matmul
    auto a = nn::Tensor::randn({ 2, 3 });
    auto b = nn::Tensor::randn({ 3, 4 });
    auto c = a.matmul(b);
    c.print("matmul result");

    // test broadcast
    auto x = nn::Tensor({ 3, 1 }, { 1, 2, 3 });
    auto y = nn::Tensor({ 1, 4 }, { 10, 20, 30, 40 });
    auto z = x + y;
    z.print("broadcast add");

    // test sum along axis
    auto m = nn::Tensor({ 2, 3 }, { 1, 2, 3, 4, 5, 6 });
    m.sum(0).print("sum axis=0");
    m.sum(1).print("sum axis=1");

    return 0;
}