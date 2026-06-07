#pragma once
#include "tensor.h"
#include "module.h"
#include <cmath>

namespace nn {
	//线性
	class Linear :public Module {
	public:
		Linear(size_t in_features, size_t out_features);
		Tensor forward(const Tensor& input)override;
		std::vector<Tensor*> parameters()override;
	private:
		Tensor W_;
		Tensor b_;
		size_t in_features_;
		size_t out_features_;
	};

	//ReLU
	class ReLU :public Module {
	public:
		Tensor forward(const Tensor& input)override;
	};
	//Sigmoid
	class Sigmoid :public Module {
	public:
		Tensor forward(const Tensor& input)override;
	};

	//Softmax
	class Softmax :public Module {
	public:
		Tensor forward(const Tensor& input)override;
	};

	class Dropout :public Module {
	public:
		Dropout(float drop_prob = 0.2);
		Tensor forward(const Tensor& input)override;
	private:
		float drop_prob_;
	};

	//BatchNorm
	class BatchNorm :public Module {
	public:
		BatchNorm(size_t num_features, float momentum = 0.1f, float epsilon = 1e-5f);
		Tensor forward(const Tensor& input)override;
		std::vector<Tensor*> parameters()override;
	private:
		size_t num_features_;
		float momentum_;
		float epsilon_;
		Tensor gamma_;
		Tensor beta_;
		Tensor running_mean_;
		Tensor running_var_;
	};

	//Sequential
	class Sequential :public Module {
	public:
		Sequential(std::vector<std::shared_ptr<Module>> layers);
		Tensor forward(const Tensor& input)override;
		std::vector<Tensor*> parameters() override;
		void train()override;
		void eval()override;
	private:
		std::vector<std::shared_ptr<Module>> layers_;
	};
	//Flatten
	class Flatten :public Module {
	public:
		Tensor forward(const Tensor& input)override;
	};
	//Conv2d
	class Conv2d : public Module {
	public:
		Conv2d(size_t in_channels, size_t out_channels, size_t kernel_size);
		Tensor forward(const Tensor& input) override;
		std::vector<Tensor*> parameters() override;
	private:
		size_t in_channels_;
		size_t out_channels_;
		size_t kernel_size_;
		Tensor weight_;
		Tensor bias_;
	};
	//MaxPool2d
	class MaxPool2d : public Module {
	public:
		MaxPool2d(size_t pool_size);
		Tensor forward(const Tensor& input) override;
	private:
		size_t pool_size_;
	};
}