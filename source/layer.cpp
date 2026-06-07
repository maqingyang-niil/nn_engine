#include "nn/layer.h"
#include "nn/autograd.h"
#include <cmath>
namespace nn {
	//Linear
	Linear::Linear(size_t in_features, size_t out_features)
		:in_features_(in_features),
		out_features_(out_features) {
		float std = std::sqrt(2.0f / (in_features + out_features));
		W_ = Tensor::randn({ in_features,out_features }) * std;
		b_ = Tensor::zeros({ out_features });

		W_.set_requires_grad(true);
		b_.set_requires_grad(true);
	}
	Tensor Linear::forward(const Tensor& input) {
		return input.matmul(W_) + b_;
	}
	std::vector<Tensor*> Linear::parameters() {
		return { &W_,&b_ };
	}
	//ReLU
	Tensor ReLU::forward(const Tensor& input) {
		Tensor result(input.shape());
		for (size_t i = 0;i < input.size();i++) {
			float val = input.data_ptr()[i];
			result.data_ptr()[i] = val > 0.0f ? val : 0.0f;
		}
		if (input.requires_grad()) {
			auto fn = std::make_shared<ReluBackward>(input);
			result.set_grad_fn(fn);
			result.set_requires_grad(true);
		}
		return result;
	}
	//Sigmoid
	Tensor Sigmoid::forward(const Tensor& input) {
		Tensor result(input.shape());
		for (size_t i = 0;i < input.size();i++) {
			result.data_ptr()[i] = 1.0f / (1.0f + std::exp(-input.data_ptr()[i]));
		}
		if (input.requires_grad()) {
			auto fn = std::make_shared<SigmoidBackward>(input, result);
			result.set_grad_fn(fn);
			result.set_requires_grad(true);
		}
		return result;
	}
	//Softmax
	Tensor Softmax::forward(const Tensor& input) {
		//为了数值安全
		Tensor max_val = input.max(1, true);
		Tensor shifted = input - max_val;
		Tensor e = shifted.exp();
		Tensor sum_e = e.sum(1, true);
		Tensor result = e / sum_e;
		if (input.requires_grad()) {
			auto fn = std::make_shared<SoftmaxBackward>(input, result);
			result.set_grad_fn(fn);
			result.set_requires_grad(true);
		}
		return result;
	}
	//Dropout
	Dropout::Dropout(float drop_prob) :drop_prob_(drop_prob) {}
	Tensor Dropout::forward(const Tensor& input) {
		if (!training_) {
			return input;
		}

		float keep_prob = 1.0f - drop_prob_;
		Tensor mask(input.shape());
		Tensor rand_vals = Tensor::rand(input.shape(), 0.0f, 1.0f);
		for (size_t i = 0;i < input.size();i++) {
			mask.data_ptr()[i] = rand_vals.data_ptr()[i] < keep_prob ? 1.0f : 0.0f;
		}
		Tensor result(input.shape());
		for (size_t i = 0;i < input.size();i++) {
			result.data_ptr()[i] = mask.data_ptr()[i] * input.data_ptr()[i] / keep_prob;
		}
		if (input.requires_grad()) {
			auto fn = std::make_shared<DropoutBackward>(input, mask, keep_prob);
			result.set_grad_fn(fn);
			result.set_requires_grad(true);
		}
		return result;
	}

	//Sequential
	Sequential::Sequential(std::vector<std::shared_ptr<Module>> layers) :layers_(layers) {}
	Tensor Sequential::forward(const Tensor& input) {
		Tensor x = input;
		for (auto& layer : layers_) {
			x = layer->forward(x);
		}
		return x;
	}

	//BatchNorm
	BatchNorm::BatchNorm(size_t num_features, float momentum, float epsilon)
		:num_features_(num_features), momentum_(momentum), epsilon_(epsilon) {
		gamma_ = Tensor::ones({ num_features_ });
		beta_ = Tensor::zeros({ num_features_ });
		running_mean_ = Tensor::zeros({ num_features });
		running_var_ = Tensor::ones({ num_features });

		gamma_.set_requires_grad(true);
		beta_.set_requires_grad(true);
	}

	Tensor BatchNorm::forward(const Tensor& input) {
		Tensor mean, var, x_hat;

		if (training_) {
			mean = input.mean(0);
			Tensor diff = input - mean;
			var = (diff * diff).mean(0);
			x_hat = diff / (var + epsilon_).sqrt();

			float* rm = running_mean_.data_ptr();
			float* rv = running_var_.data_ptr();
			const float* m = mean.data_ptr();
			const float* v = var.data_ptr();
			for (size_t i = 0; i < num_features_; i++) {
				rm[i] = (1.0f - momentum_) * rm[i] + momentum_ * m[i];
				rv[i] = (1.0f - momentum_) * rv[i] + momentum_ * v[i];
			}
		}
		else {
			x_hat = (input - running_mean_) / (running_var_ + epsilon_).sqrt();
		}
		return gamma_ * x_hat + beta_;
	}

	std::vector<Tensor*> BatchNorm::parameters() {
		return { &gamma_,&beta_ };
	}

	//Sequential
	std::vector<Tensor*> Sequential::parameters() {
		std::vector<Tensor*> params;
		for (auto& layer : layers_) {
			auto layer_params = layer->parameters();
			params.insert(params.end(), layer_params.begin(), layer_params.end());
		}
		return params;
	}

	void Sequential::train() {
		training_ = true;
		for (auto& layer : layers_) {
			layer->train();
		}
	}

	void Sequential::eval() {
		training_ = false;
		for (auto& layer : layers_) {
			layer->eval();
		}
	}

	//Flatten
	Tensor Flatten::forward(const Tensor& input) {
		size_t batch = input.shape()[0];
		size_t feature = input.size() / batch;
		Tensor result = input.reshape({ batch,feature });

		if (input.requires_grad()) {
			auto fn = std::make_shared<FlattenBackward>(input);
			result.set_grad_fn(fn);
			result.set_requires_grad(true);
		}
		return result;
	}

	//Conv2d
	Conv2d::Conv2d(size_t in_channels, size_t out_channels, size_t kernel_size)
		: in_channels_(in_channels), out_channels_(out_channels), kernel_size_(kernel_size) {
		float std_val = std::sqrt(2.0f / (in_channels * kernel_size * kernel_size));
		weight_ = Tensor::randn({ out_channels, in_channels, kernel_size, kernel_size }) * std_val;
		bias_ = Tensor::zeros({ out_channels });
		weight_.set_requires_grad(true);
		bias_.set_requires_grad(true);
	}

	Tensor Conv2d::forward(const Tensor& input) {
		size_t batch = input.shape()[0];
		size_t outH = input.shape()[2] - kernel_size_ + 1;
		size_t outW = input.shape()[3] - kernel_size_ + 1;

		Tensor col = Tensor::im2col(input, kernel_size_, kernel_size_);
		Tensor weight_2d = weight_.reshape({ out_channels_, in_channels_ * kernel_size_ * kernel_size_ });
		Tensor out_2d = col.matmul(weight_2d.transpose(0, 1));
		out_2d = out_2d + bias_;

		// reshape成 (batch, outH, outW, out_channels)，再permute成 (batch, out_channels, outH, outW)
		std::vector<float> permuted(batch * out_channels_ * outH * outW);
		const float* src = out_2d.data_ptr();
		for (size_t n = 0; n < batch; n++) {
			for (size_t oh = 0; oh < outH; oh++) {
				for (size_t ow = 0; ow < outW; ow++) {
					for (size_t oc = 0; oc < out_channels_; oc++) {
						permuted[n * out_channels_ * outH * outW + oc * outH * outW + oh * outW + ow]
							= src[n * outH * outW * out_channels_ + oh * outW * out_channels_ + ow * out_channels_ + oc];
					}
				}
			}
		}
		Tensor result({ batch, out_channels_, outH, outW }, permuted);

		if (input.requires_grad()) {
			auto fn = std::make_shared<Conv2dBackward>(input, weight_, col, kernel_size_, kernel_size_);
			result.set_grad_fn(fn);
			result.set_requires_grad(true);
		}

		return result;
	}

	std::vector<Tensor*> Conv2d::parameters() {
		return { &weight_, &bias_ };
	}

	//MaxPool2d
	MaxPool2d::MaxPool2d(size_t pool_size) : pool_size_(pool_size) {}

	Tensor MaxPool2d::forward(const Tensor& input) {
		size_t batch = input.shape()[0];
		size_t C = input.shape()[1];
		size_t H = input.shape()[2];
		size_t W = input.shape()[3];
		size_t outH = H / pool_size_;
		size_t outW = W / pool_size_;

		std::vector<float> output(batch * C * outH * outW);
		std::vector<size_t> max_indices(batch * C * outH * outW);

		for (size_t n = 0; n < batch; n++) {
			for (size_t c = 0; c < C; c++) {
				for (size_t oh = 0; oh < outH; oh++) {
					for (size_t ow = 0; ow < outW; ow++) {
						float max_val = -1e30f;
						size_t max_idx = 0;
						for (size_t ph = 0; ph < pool_size_; ph++) {
							for (size_t pw = 0; pw < pool_size_; pw++) {
								size_t ih = oh * pool_size_ + ph;
								size_t iw = ow * pool_size_ + pw;
								size_t idx = n * C * H * W + c * H * W + ih * W + iw;
								if (input.data_ptr()[idx] > max_val) {
									max_val = input.data_ptr()[idx];
									max_idx = idx;
								}
							}
						}
						size_t out_idx = n * C * outH * outW + c * outH * outW + oh * outW + ow;
						output[out_idx] = max_val;
						max_indices[out_idx] = max_idx;
					}
				}
			}
		}

		Tensor result({ batch, C, outH, outW }, output);

		if (input.requires_grad()) {
			auto fn = std::make_shared<MaxPool2dBackward>(input, max_indices, pool_size_);
			result.set_grad_fn(fn);
			result.set_requires_grad(true);
		}

		return result;
	}

}