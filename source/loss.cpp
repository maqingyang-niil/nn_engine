#include "nn/loss.h"

namespace nn {
	Tensor MSELoss::forward(const Tensor& pred, const Tensor& target) {
		Tensor diff = pred - target;
		Tensor sq = diff.pow(2.0f);
		float n = static_cast<float>(pred.size());
		return sq.sum() / n;
	}

	Tensor CrossEntropyLoss::forward(const Tensor& pred, const Tensor& target) {
		float epsilon = 1e-7f;
		Tensor safe_pred = pred + epsilon;
		Tensor log_pred = safe_pred.log();
		Tensor loss = -1.0f * (target * log_pred).sum() / static_cast<float>(pred.shape()[0]);
		return loss;
	}
}