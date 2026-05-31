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

	class SubBackward :public Autograd {
	public:
		SubBackward(const Tensor& a, const Tensor& b) {
			saved_inputs.push_back(a);
			saved_inputs.push_back(b);
		}

		std::vector<Tensor> backward(const Tensor& grad_output)override {
			return{
				grad_output,
				-grad_output
			};
		}
	};

	class NegBackward :public Autograd {
	public:
		NegBackward(const Tensor& a) {
			saved_inputs.push_back(a);
		}

		std::vector<Tensor> backward(const Tensor& grad_output)override {
			return{ -grad_output };
		}
	};

	class DivBackward :public Autograd {
	public:
		DivBackward(const Tensor& a, const Tensor& b) {
			saved_inputs.push_back(a);
			saved_inputs.push_back(b);
		}

		std::vector<Tensor> backward(const Tensor& grad_output)override {
			return{
				grad_output / saved_inputs[1],
				-1 * grad_output * saved_inputs[0] / saved_inputs[1] / saved_inputs[1]
			};
		}
	};

	class MulScalarBackward :public Autograd {
	public:
		float scalar_;
		MulScalarBackward(float scalar, const Tensor& a) :scalar_(scalar) {
			saved_inputs.push_back(a);
		}

		std::vector<Tensor> backward(const Tensor& grad_output)override {
			return { grad_output * scalar_ };
		}
	};

	class AddScalarBackward :public Autograd {
	public:
		float scalar_;
		AddScalarBackward(float scalar, const Tensor& a) :scalar_(scalar) {
			saved_inputs.push_back(a);
		}

		std::vector<Tensor> backward(const Tensor& grad_output)override {
			return { grad_output };
		}
	};

	class LSubScalarBackward :public Autograd {
	public:
		float scalar_;
		LSubScalarBackward(float scalar, const Tensor& a) :scalar_(scalar) {
			saved_inputs.push_back(a);
		}

		std::vector<Tensor> backward(const Tensor& grad_output)override {
			return { grad_output };
		}
	};

	class LDivScalarBackward :public Autograd {
	public:
		float scalar_;
		LDivScalarBackward(float scalar, const Tensor& a) :scalar_(scalar) {
			saved_inputs.push_back(a);
		}

		std::vector<Tensor> backward(const Tensor& grad_output)override {
			return { grad_output / scalar_ };
		}
	};

	class RSubScalarBackward :public Autograd {
	public:
		float scalar_;
		RSubScalarBackward(float scalar, const Tensor& a) :scalar_(scalar) {
			saved_inputs.push_back(a);
		}
		std::vector<Tensor> backward(const Tensor& grad_output)override {
			return{ -grad_output };
		}
	};

	class RDivScalarBackward :public Autograd {
	public:
		float scalar_;
		RDivScalarBackward(float scalar, const Tensor& a) :scalar_(scalar) {
			saved_inputs.push_back(a);
		}

		std::vector<Tensor> backward(const Tensor& grad_output)override {
			return { -grad_output * scalar_ / saved_inputs[0] / saved_inputs[0] };
		}
	};

	class MatMulBackward :public Autograd {
	public:
		MatMulBackward(const Tensor& a, const Tensor& b) {
			saved_inputs.push_back(a);
			saved_inputs.push_back(b);
		}

		std::vector<Tensor> backward(const Tensor& grad_output)override {
			return {
				grad_output.matmul(saved_inputs[1].transpose()),
				saved_inputs[0].transpose().matmul(grad_output)
			};
		}
	};

	class SumBackward :public Autograd {
	public:
		std::vector<size_t> input_shape_;
		SumBackward(const Tensor& a) :input_shape_(a.shape()) {
			saved_inputs.push_back(a);
		}
		std::vector<Tensor> backward(const Tensor& grad_output)override {
			return { Tensor::full(input_shape_,grad_output[0]) };
		}
	};

	class ExpBackward :public Autograd {
	public:
		Tensor output_;
		ExpBackward(const Tensor& input, const Tensor& output):output_(output) {
			saved_inputs.push_back(input);
		}
		std::vector<Tensor> backward(const Tensor& grad_output)override {
			return { grad_output * output_ };
		}
	};

	class LogBackward :public Autograd {
	public:
		LogBackward(const Tensor& a) {
			saved_inputs.push_back(a);
		}

		std::vector<Tensor> backward(const Tensor& grad_output)override {
			return { grad_output / saved_inputs[0] };
		}
	};

	class PowBackward :public Autograd {
	public:
		float exponent_;
		PowBackward(float exponent, const Tensor& a) :exponent_(exponent) {
			saved_inputs.push_back(a);
		}
		std::vector<Tensor> backward(const Tensor& grad_output)override {
			return { grad_output * exponent_ * saved_inputs[0].pow(exponent_ - 1) };
		}
	};
}