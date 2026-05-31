#include "nn/tensor.h"
#include "nn/autograd.h"
#include <iostream>

using namespace nn;

void test(const std::string& name, float got, float expected) {
    bool pass = std::abs(got - expected) < 1e-4f;
    std::cout << (pass ? "[PASS] " : "[FAIL] ") << name
        << " got=" << got << " expected=" << expected << "\n";
}

int main() {
    // --- SubBackward: loss = a - b, dL/da=1, dL/db=-1 ---
    {
        Tensor a({ 1 }, { 5.0f }), b({ 1 }, { 3.0f });
        a.set_requires_grad(true); b.set_requires_grad(true);
        Tensor loss = a - b;
        loss.backward();
        test("Sub grad_a", (*a.grad())[0], 1.0f);
        test("Sub grad_b", (*b.grad())[0], -1.0f);
    }

    // --- NegBackward: loss = -a, dL/da=-1 ---
    {
        Tensor a({ 1 }, { 4.0f });
        a.set_requires_grad(true);
        Tensor loss = -a;
        loss.backward();
        test("Neg grad_a", (*a.grad())[0], -1.0f);
    }

    // --- DivBackward: loss = a / b, dL/da=1/b, dL/db=-a/b^2 ---
    {
        Tensor a({ 1 }, { 6.0f }), b({ 1 }, { 2.0f });
        a.set_requires_grad(true); b.set_requires_grad(true);
        Tensor loss = a / b;
        loss.backward();
        test("Div grad_a", (*a.grad())[0], 0.5f);    // 1/2
        test("Div grad_b", (*b.grad())[0], -1.5f);    // -6/4
    }

    // --- AddScalarBackward: loss = a + 3, dL/da=1 ---
    {
        Tensor a({ 1 }, { 2.0f });
        a.set_requires_grad(true);
        Tensor loss = a + 3.0f;
        loss.backward();
        test("AddScalar grad_a", (*a.grad())[0], 1.0f);
    }

    // --- MulScalarBackward: loss = a * 4, dL/da=4 ---
    {
        Tensor a({ 1 }, { 2.0f });
        a.set_requires_grad(true);
        Tensor loss = a * 4.0f;
        loss.backward();
        test("MulScalar grad_a", (*a.grad())[0], 4.0f);
    }

    // --- DivScalarBackward: loss = a / 4, dL/da=0.25 ---
    {
        Tensor a({ 1 }, { 8.0f });
        a.set_requires_grad(true);
        Tensor loss = a / 4.0f;
        loss.backward();
        test("DivScalar grad_a", (*a.grad())[0], 0.25f);
    }

    // --- RSubScalarBackward: loss = 5 - a, dL/da=-1 ---
    {
        Tensor a({ 1 }, { 2.0f });
        a.set_requires_grad(true);
        Tensor loss = 5.0f - a;
        loss.backward();
        test("RSub grad_a", (*a.grad())[0], -1.0f);
    }

    // --- RDivScalarBackward: loss = 6 / a, dL/da=-6/a^2 ---
    {
        Tensor a({ 1 }, { 3.0f });
        a.set_requires_grad(true);
        Tensor loss = 6.0f / a;
        loss.backward();
        test("RDiv grad_a", (*a.grad())[0], -6.0f / 9.0f);  // -6/9
    }

    // --- SumBackward: loss = sum([a,a,a]), dL/da=[1,1,1] ---
    {
        Tensor a({ 3 }, { 1.0f, 2.0f, 3.0f });
        a.set_requires_grad(true);
        Tensor loss = a.sum();
        loss.backward();
        test("Sum grad_a[0]", (*a.grad())[0], 1.0f);
        test("Sum grad_a[1]", (*a.grad())[1], 1.0f);
        test("Sum grad_a[2]", (*a.grad())[2], 1.0f);
    }

    // --- ExpBackward: loss = exp(a), dL/da=exp(a) ---
    {
        Tensor a({ 1 }, { 2.0f });
        a.set_requires_grad(true);
        Tensor loss = a.exp();
        loss.backward();
        test("Exp grad_a", (*a.grad())[0], std::exp(2.0f));
    }

    // --- LogBackward: loss = log(a), dL/da=1/a ---
    {
        Tensor a({ 1 }, { 4.0f });
        a.set_requires_grad(true);
        Tensor loss = a.log();
        loss.backward();
        test("Log grad_a", (*a.grad())[0], 0.25f);  // 1/4
    }

    // --- PowBackward: loss = a^3, dL/da=3*a^2 ---
    {
        Tensor a({ 1 }, { 2.0f });
        a.set_requires_grad(true);
        Tensor loss = a.pow(3.0f);
        loss.backward();
        test("Pow grad_a", (*a.grad())[0], 12.0f);  // 3*4=12
    }

    // --- MatMulBackward: C=A@B, dL/dA=dL/dC@B.T, dL/dB=A.T@dL/dC ---
    {
        Tensor A({ 2,2 }, { 1,2,3,4 });
        Tensor B({ 2,2 }, { 5,6,7,8 });
        A.set_requires_grad(true); B.set_requires_grad(true);
        Tensor C = A.matmul(B);
        Tensor loss = C.sum();
        loss.backward();
        // dL/dC = ones(2,2)
        // dL/dA = ones @ B.T = [[11,15],[11,15]]
        // dL/dB = A.T @ ones = [[4,4],[6,6]]
        test("MatMul grad_A[0]", (*A.grad())[0], 11.0f);
        test("MatMul grad_A[1]", (*A.grad())[1], 15.0f);
        test("MatMul grad_B[0]", (*B.grad())[0], 4.0f);
        test("MatMul grad_B[2]", (*B.grad())[2], 6.0f);
    }
}