#include "nn/optimizer.h"
#include "nn/autograd.h"

namespace nn {
	//SGD
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
	//Adam
	Adam::Adam(std::vector<Tensor*> params, float lr, float beta1,
		float beta2, float epsilon) :params_(params), lr_(lr), beta1_(beta1),
		beta2_(beta2), epsilon_(epsilon), t_(0) {
		for (auto* p : params_) {
			m_.push_back(Tensor::zeros(p->shape()));
			v_.push_back(Tensor::zeros(p->shape()));
		}
	}
	void Adam::step() {
		t_++;
		for (size_t i = 0;i < params_.size();i++) {
			auto* param = params_[i];
			if (!param->grad()) continue;

			Tensor grad = *param->grad();
			m_[i] = beta1_ * m_[i] + (1.0f - beta1_) * grad;
			v_[i] = beta2_ * v_[i] + (1.0f - beta2_) * grad * grad;

			float bias_correction1 = 1.0f - std::pow(beta1_, t_);
			float bias_correction2 = 1.0f - std::pow(beta2_, t_);

			Tensor m_hat = m_[i] / bias_correction1;
			Tensor v_hat = v_[i] / bias_correction2;

			Tensor updated= *param - lr_ * m_hat / (v_hat.sqrt() + epsilon_);

			for (size_t j = 0;j < param->size();j++) {
				param->data_ptr()[j] = updated.data_ptr()[j];
			}
		}
		
	}
	void Adam::zero_grad() {
		for (auto* param : params_) {
			param->set_grad(nullptr);
		}
	}
}