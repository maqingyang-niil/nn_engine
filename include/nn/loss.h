#pragma once
#include "tensor.h"
#include "module.h"

namespace nn {
	class MSELoss {
	public:
		Tensor forward(const Tensor& pred, const Tensor& target);
	};

	class CrossEntropyLoss {
	public:
		Tensor forward(const Tensor& pred, const Tensor& target);
	};
}