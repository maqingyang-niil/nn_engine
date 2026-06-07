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
	
	//乘法求导
	class MulBackward :public Autograd {
	public:
		MulBackward(const Tensor& a, const Tensor& b) {
			saved_inputs.push_back(a);
			saved_inputs.push_back(b);
		}

		std::vector<Tensor> backward(const Tensor& grad_output) override {
			return {
				reduce_to_shape(grad_output * saved_inputs[1],saved_inputs[0].shape()),
				reduce_to_shape(grad_output * saved_inputs[0],saved_inputs[1].shape())
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
			return {
				reduce_to_shape(grad_output,saved_inputs[0].shape()),
				reduce_to_shape(-grad_output,saved_inputs[1].shape())
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
			return {
				reduce_to_shape(grad_output / saved_inputs[1],saved_inputs[0].shape()),
				reduce_to_shape(-grad_output * saved_inputs[0] / saved_inputs[1] / saved_inputs[1],saved_inputs[1].shape())
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

	//Softmax求导
	class SoftmaxBackward :public Autograd {
	public:
		Tensor output_;
		SoftmaxBackward(const Tensor& input, const Tensor& output) :output_(output) {
			saved_inputs.push_back(input);
		}
		std::vector<Tensor> backward(const Tensor& grad_output)override {
			Tensor sg = grad_output * output_;
			Tensor sum_sg = sg.sum(1, true);
			return { output_ * (grad_output - sum_sg) };
		}
	};

	//Dropout求导
	class DropoutBackward :public Autograd {
	public:
		Tensor mask_;
		float keep_prob_;
		DropoutBackward(const Tensor& input, const Tensor& mask, float keep_prob)
			:mask_(mask), keep_prob_(keep_prob) {
			saved_inputs.push_back(input);
		}

		std::vector<Tensor> backward(const Tensor& grad_output)override {
			return { grad_output * mask_ / keep_prob_ };
		}
	};

	//Flatten求导
	class FlattenBackward :public Autograd {
	public:
		std::vector<size_t> original_shape_;
		FlattenBackward(const Tensor& input) :original_shape_(input.shape()) {
			saved_inputs.push_back(input);
		}
		std::vector<Tensor> backward(const Tensor& grad_output)override {
			return { grad_output.reshape(original_shape_) };
		}
	};

	//ConV2d
	class Conv2dBackward :public Autograd {
	public:
		Tensor col_;
		Tensor weight_;
		std::vector<size_t> input_shape_;
		size_t kH_, kW_;
		Conv2dBackward(const Tensor& input, const Tensor& weight, const Tensor& col, size_t kH, size_t kW)
			:col_(col), weight_(weight), input_shape_(input.shape()), kH_(kH), kW_(kW) {
			saved_inputs.push_back(input);
			saved_inputs.push_back(weight);
		}
		std::vector<Tensor> backward(const Tensor& grad_output) override {
			size_t batch = grad_output.shape()[0];
			size_t out_channels = grad_output.shape()[1];
			size_t outH = grad_output.shape()[2];
			size_t outW = grad_output.shape()[3];
			size_t col_cols = col_.shape()[1];

			// grad_output: (batch, out_channels, outH, outW)
			// 先手动permute成 (batch, outH, outW, out_channels)
			// 再reshape成 (batch*outH*outW, out_channels)
			std::vector<float> permuted(batch * out_channels * outH * outW);
			const float* g = grad_output.data_ptr();
			for (size_t n = 0; n < batch; n++) {
				for (size_t oc = 0; oc < out_channels; oc++) {
					for (size_t oh = 0; oh < outH; oh++) {
						for (size_t ow = 0; ow < outW; ow++) {
							permuted[n * outH * outW * out_channels + oh * outW * out_channels + ow * out_channels + oc]
								= g[n * out_channels * outH * outW + oc * outH * outW + oh * outW + ow];
						}
					}
				}
			}
			Tensor grad_out_2d({ batch * outH * outW, out_channels }, permuted);

			// 权重梯度: col^T × grad_out_2d → (col_cols, out_channels) → 转置再reshape
			Tensor col_T = col_.transpose(0, 1);
			Tensor grad_weight = col_T.matmul(grad_out_2d).transpose(0, 1).reshape(weight_.shape());

			// 输入梯度: grad_out_2d × weight_2d → col2im还原
			Tensor weight_2d = weight_.reshape({ out_channels, col_cols });
			Tensor grad_col = grad_out_2d.matmul(weight_2d);
			Tensor grad_input = Tensor::col2im(grad_col, input_shape_, kH_, kW_);

			// 偏置梯度: grad_out_2d 按第0维求和 → (out_channels,)
			Tensor grad_bias = grad_out_2d.sum(0);

			return { grad_input, grad_weight, grad_bias };
		}
	};

	//MaxPool2d
	class MaxPool2dBackward : public Autograd {
	public:
		std::vector<size_t> max_indices_;
		std::vector<size_t> input_shape_;
		size_t pool_size_;

		MaxPool2dBackward(const Tensor& input, const std::vector<size_t>& max_indices, size_t pool_size)
			: max_indices_(max_indices), input_shape_(input.shape()), pool_size_(pool_size) {
			saved_inputs.push_back(input);
		}

		std::vector<Tensor> backward(const Tensor& grad_output) override {
			std::vector<float> grad_input(grad_output.size() * pool_size_ * pool_size_, 0.0f);
			const float* g = grad_output.data_ptr();
			for (size_t i = 0; i < grad_output.size(); i++) {
				grad_input[max_indices_[i]] = g[i];
			}
			return { Tensor(input_shape_, grad_input) };
		}
	};
}