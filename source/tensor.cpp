#include "nn/tensor.h"

namespace nn {
	
	Tensor::Tensor()
		:data_(std::make_shared<std::vector<float>>())
		, shape_({})
		, strides_({})
		, offset_(0){}

	//从shape构造
	Tensor::Tensor(const std::vector<size_t>& shape)
		:shape_(shape)
		, offset_(0) {
		size_t total = 1;
		for (auto s : shape)total *= s;
		data_ = std::make_shared<std::vector<float>>(total, 0.0f);
		compute_strides();
	}

	Tensor::Tensor(const std::vector<size_t>& shape, const std::vector<float>& data)
		:shape_(shape)
		, offset_(0) {
		size_t total = 1;
		for (auto s : shape_)total *= s;
		if (data.size() != total) {
			throw std::invalid_argument(
				"Data size(" + std::to_string(data.size()) +
				")does not match shape (total=" + std::to_string(total) + ")"
			);
		}
		data_ = std::make_shared<std::vector<float>>(data);
		compute_strides();
	}

	Tensor::Tensor(const std::vector<size_t>& shape, const std::initializer_list<float>& data)
		:Tensor(shape, std::vector<float>(data)) {}

	//拷贝构造（浅拷贝，共享数据）
	Tensor::Tensor(const Tensor& other)
		:data_(other.data_)
		, shape_(other.shape_)
		, strides_(other.strides_)
		, offset_(other.offset_){ }

	//移动构造
	Tensor::Tensor(Tensor&& other) noexcept
		:data_(std::move(other.data_))
		, shape_(std::move(other.shape_))
		, strides_(std::move(other.strides_))
		, offset_(other.offset_) {
		other.offset_ = 0;//对象处于安全状态的显示处理
	}

	//拷贝赋值
	Tensor& Tensor::operator=(const Tensor& other) {
		if (this != &other) {
			data_ = other.data_;
			shape_ = other.shape_;
			strides_ = other.strides_;
			offset_ = other.offset_;
		}
		return *this;
	}

	//移动赋值
	Tensor& Tensor::operator=(Tensor&& other) {
		if (this != &other) {
			data_ = std::move(other.data_);
			shape_ = std::move(other.shape_);
			strides_ = std::move(other.strides_);
			offset_ = other.offset_;
			other.offset_ = 0;
		}
		return *this;
	}

	//全零tensor
	Tensor Tensor::zeros(const std::vector<size_t>& shape) {
		return Tensor(shape);
	}

	//全一
	Tensor Tensor::ones(const std::vector<size_t>& shape) {
		Tensor t(shape);
		t.fill(1.0f);
		return t;
	}

	//指定位填充
	Tensor Tensor::full(const std::vector<size_t>& shape, float value) {
		Tensor t(shape);
		t.fill(value);
		return t;
	}

	//标准正太分布随机初始化
	Tensor Tensor::randn(const std::vector<size_t>& shape) {
		Tensor t(shape);
		static std::mt19937 gen(std::random_device{}());
		std::normal_distribution<float> dist(0.0f, 1.0f);
		for (size_t i = 0;i < t.size();i++) {
			(*t.data_)[i] = dist(gen);
		}
		return t;
	}

	// 均匀分布随机初始化 [low, high)
	Tensor Tensor::rand(const std::vector<size_t>& shape, float low = 0.0f, float high = 1.0f) {
		Tensor t(shape);
		static std::mt19937 gen(std::random_device{}());
		std::uniform_real_distribution<float> dist(low, high);
		for (size_t i = 0;i < t.size();i++) {
			(*t.data_)[i] = dist(gen);
		}
		return t;
	}

	// 单位矩阵
	Tensor Tensor::eye(size_t n) {
		Tensor t({ n,n });
		for (size_t i = 0;i < n;i++) {
			t.at({ i,i }) = 1.0f;
		}
		return t;
	}

	//形状
	const std::vector<size_t>& Tensor::shape()const {
		return shape_;
	}

	//步长
	const std::vector<size_t>& Tensor::strides()const {
		return strides_;
	}

	//维度
	size_t Tensor::ndim()const {
		return shape_.size();
	}


	//计算步长
	void Tensor::compute_strides() {
		strides_.resize(shape_.size());
		if (shape_.empty())return;
		strides_.back() = 1;
		for (int i = static_cast<int>(shape_.size()) - 2;i >= 0;i--) {
			strides_[i] = strides_[i + 1] * shape_[i + 1];
		}
	}

}