#include "nn/optimizer.h"
#include "nn/autograd.h"

namespace nn {
	SGD::SGD(std::vector<Tensor*> params, float lr)
		:params_(params), lr_(lr){ }

	void SGD::step() {
		for (auto* param : params_) {
			if (!param->grad()) continue;
			Tensor updated = *param - lr_ * (*param->grad());
			for (size_t i = 0;i < param->size();i++) {
				param->data_ptr()[i] = updated.data_ptr()[i];
			}
		}
	}

	void SGD::zero_grad() {
		for (auto* param : params_) {
			param->set_grad(nullptr);
		}
	}
}