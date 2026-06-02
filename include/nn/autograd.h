#pragma once
#include "tensor.h"
#include <vector>

namespace nn {

	inline Tensor reduce_to_shape(const Tensor& grad, 
		                          const std::vector<size_t>& target) {
		Tensor result = grad;
		while (result.ndim() > target.size()) {
			result = result.sum(0);
		}
		for (size_t i = 0;i < target.size();i++) {
			if (target[i] == 1 && result.shape()[i] > 1)
				result = result.sum(i, true);
		}
		return result;
	}

	class Autograd {
	public:
		//读取前向计算时的输入
		const std::vector<Tensor>& inputs() const {
			return saved_inputs;
		}
		//子类实现求导公式
		virtual std::vector<Tensor> backward(const Tensor& grad_output) = 0;

		virtual ~Autograd() = default;

	protected:
		//前向计算时的输入
		std::vector<Tensor> saved_inputs;
		
	};

	class AddBackward :public Autograd {
	public:
		AddBackward(const Tensor& a, const Tensor& b) {
			saved_inputs.push_back(a);
			saved_inputs.push_back(b);
		}

		std::vector<Tensor> backward(const Tensor& grad_output) override {
			return {
				reduce_to_shape(grad_output,saved_inputs[0].shape()),
				reduce_to_shape(grad_output,saved_inputs[1].shape())
			};
		}
	};
	
	//加法求导
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

	//减法求导
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

	//取负求导
	class NegBackward :public Autograd {
	public:
		NegBackward(const Tensor& a) {
			saved_inputs.push_back(a);
		}

		std::vector<Tensor> backward(const Tensor& grad_output)override {
			return{ -grad_output };
		}
	};

	//除法求导
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

	//变量*Tensor 求导
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

	//标量＋Tensor 求导
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

	//Tensor-标量求导
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

	//Tensor÷标量求导
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

	//标量-Tensor求导
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

	//标量÷Tensor求导（除了matmul，都是逐元素运算）
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

	//Tensor×Tensor求导
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

	//求和求导
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

	/*
	指数求导
	同时保存输入和输出，输出用于在求导过程中减少计算，输入用于确定反向传播的方向
	*/
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

	/*
	对数求导
	*/
	class LogBackward :public Autograd {
	public:
		LogBackward(const Tensor& a) {
			saved_inputs.push_back(a);
		}

		std::vector<Tensor> backward(const Tensor& grad_output)override {
			return { grad_output / saved_inputs[0] };
		}
	};

	//幂数求导
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

	//Relu求导
	class ReluBackward :public Autograd {
	public:
		ReluBackward(const Tensor& a) {
			saved_inputs.push_back(a);
		}
		std::vector<Tensor> backward(const Tensor& grad_output)override {
			Tensor mask = saved_inputs[0] > 0.0f;
			return { grad_output * mask };
		}
	};

	//sigmoid求导
	class SigmoidBackward :public Autograd {
	public:
		Tensor output_;
		SigmoidBackward(const Tensor& input,const Tensor& output):output_(output) {
			saved_inputs.push_back(input);
		}
		std::vector<Tensor> backward(const Tensor& grad_output)override {
			return { grad_output * output_ * (1.0f - output_) };
		}
	};
}