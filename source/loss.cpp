#include "nn/loss.h"

namespace nn {
	Tensor MSELoss::forward(const Tensor& pred, const Tensor& target) {
		Tensor diff = pred - target;
		Tensor sq = diff.pow(2.0f);
		float n = static_cast<float>(pred.size());
		return sq.sum() / n;
	}
}