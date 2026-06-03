#pragma once
#include <vector>
#include "tensor.h"


namespace nn {
	class SGD {
	public:
		SGD(std::vector<Tensor*> params, float lr);
		void step();
		void zero_grad();
	private:
		std::vector<Tensor*> params_;
		float lr_;
	};

	class Adam {
	public:
		Adam(std::vector<Tensor*> params, float lr = 0.001f, float beta1 = 0.9f,
			float beta2 = 0.999f, float epsilon = 1e-8f);
		void step();
		void zero_grad();
	private:
		std::vector<Tensor*> params_;
		float lr_;
		float beta1_;
		float beta2_;
		float epsilon_;
		int t_;
		std::vector<Tensor> m_;
		std::vector<Tensor> v_;
	};
}