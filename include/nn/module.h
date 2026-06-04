#pragma once
#include "tensor.h"
#include <vector>
namespace nn{
	class Module {
	public:
		virtual Tensor forward(const Tensor& input) = 0;
		virtual std::vector<Tensor*> parameters() { return{}; }
		virtual void train() { training_ = true; }
		virtual void eval() { training_ = false; }
		virtual ~Module() = default;
	protected:
		bool training_ = true;
    };
}