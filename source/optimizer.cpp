#include "nn/optimizer.h"
#include "nn/autograd.h"

namespace nn {
	//SGD
	SGD::SGD(std::vector<Tensor*> params, float lr)
		:params_(params), lr_(lr){ }

	void SGD::step() {
		for (auto* param : params_) {
			if (!param->grad())continue;
			for (size_t i = 0;i < param->size();i++) {
				param->data_ptr()[i] = param->data_ptr()[i] - lr_ * (param->grad()->data_ptr()[i]);
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
		float bias_correction1 = 1.0f - std::pow(beta1_, t_);
		float bias_correction2 = 1.0f - std::pow(beta2_, t_);

		for (size_t i = 0;i < params_.size();i++) {
			auto* param = params_[i];
			if (!param->grad()) continue;

			for (size_t j = 0;j < param->size();j++) {
				m_[i].data_ptr()[j] = beta1_ * m_[i].data_ptr()[j] + (1.0f - beta1_) * (param->grad()->data_ptr()[j]);
				v_[i].data_ptr()[j] = beta2_ * v_[i].data_ptr()[j] + (1.0f - beta2_) * (param->grad()->data_ptr()[j]) * (param->grad()->data_ptr()[j]);
				float m_hat = m_[i].data_ptr()[j] / bias_correction1;
				float v_hat = v_[i].data_ptr()[j] / bias_correction2;
				param->data_ptr()[j] -= lr_ * m_hat / (std::sqrt(v_hat) + epsilon_);
			}
		}
		
	}

	void Adam::zero_grad() {
		for (auto* param : params_) {
			param->set_grad(nullptr);
		}
	}
}