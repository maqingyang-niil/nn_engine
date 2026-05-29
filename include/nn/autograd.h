#pragma once
#include "tensor.h"
#include <vector>

namespace nn {
	class Autograd {
	public:
		const std::vector<Tensor>& inputs() const {
			return saved_inputs;
		}
		virtual std::vector<Tensor> backward(const Tensor& grad_output) = 0;
		virtual ~Autograd() = default;

	protected:
		std::vector<Tensor> saved_inputs;
		
	};

	class AddBackward :public Autograd {
	public:
		AddBackward(const Tensor& a, const Tensor& b) {
			saved_inputs.push_back(a);
			saved_inputs.push_back(b);
		}

		std::vector<Tensor> backward(const Tensor& grad_output) override {
			// 对于加法，输入的梯度就是输出的梯度
			return { grad_output, grad_output };
		}
	};

	class MulBackward :public Autograd {
	public:
		MulBackward(const Tensor& a, const Tensor& b) {
			saved_inputs.push_back(a);
			saved_inputs.push_back(b);
		}

		std::vector<Tensor> backward(const Tensor& grad_output) override {
			return {
				grad_output * saved_inputs[1],
				grad_output * saved_inputs[0]
			};
		}
	};
}