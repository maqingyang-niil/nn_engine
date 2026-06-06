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

}