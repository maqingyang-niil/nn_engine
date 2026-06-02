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
}