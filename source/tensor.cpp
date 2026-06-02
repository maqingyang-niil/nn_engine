#include "nn/tensor.h"
#include "nn/autograd.h"
#include <unordered_set>

namespace nn {
	//构造
	Tensor::Tensor()
		:data_(std::make_shared<std::vector<float>>())
		, shape_({})
		, strides_({})
		, offset_(0)
	    , grad_(std::make_shared<std::shared_ptr<Tensor>>(nullptr)){}

	//从shape构造
	Tensor::Tensor(const std::vector<size_t>& shape)
		:shape_(shape)
		, offset_(0) 
		, grad_(std::make_shared<std::shared_ptr<Tensor>>(nullptr)) {
		size_t total = 1;
		for (auto s : shape)total *= s;
		data_ = std::make_shared<std::vector<float>>(total, 0.0f);
		compute_strides();
	}
	//从shape和元素构造
	Tensor::Tensor(const std::vector<size_t>& shape, const std::vector<float>& data)
		:shape_(shape)
		, offset_(0) 
		, grad_(std::make_shared<std::shared_ptr<Tensor>>(nullptr)) {
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

	//委托构造（花括号包围数据）
	Tensor::Tensor(const std::vector<size_t>& shape, const std::initializer_list<float>& data)
		:Tensor(shape, std::vector<float>(data)) {}

	//拷贝构造（浅拷贝，共享数据）
	Tensor::Tensor(const Tensor& other)
		:data_(other.data_)
		, shape_(other.shape_)
		, strides_(other.strides_)
		, offset_(other.offset_)
	    , requires_grad_(other.requires_grad_)
	    , grad_(other.grad_)
	    , grad_fn_(other.grad_fn_){ }

	//移动构造
	Tensor::Tensor(Tensor&& other) noexcept
		:data_(std::move(other.data_))
		, shape_(std::move(other.shape_))
		, strides_(std::move(other.strides_))
		, offset_(other.offset_) 
	    , requires_grad_(other.requires_grad_)
	    , grad_(std::move(other.grad_))
	    , grad_fn_(std::move(other.grad_fn_)){
		other.offset_ = 0;//对象处于安全状态的显示处理
	}

	//拷贝赋值
	Tensor& Tensor::operator=(const Tensor& other) {
		if (this != &other) {
			data_ = other.data_;
			shape_ = other.shape_;
			strides_ = other.strides_;
			offset_ = other.offset_;
			requires_grad_ = other.requires_grad_;
			grad_ = other.grad_;
			grad_fn_ = other.grad_fn_;

		}
		return *this;
	}

	//移动赋值
	Tensor& Tensor::operator=(Tensor&& other) noexcept {
		if (this != &other) {
			data_ = std::move(other.data_);
			shape_ = std::move(other.shape_);
			strides_ = std::move(other.strides_);
			offset_ = other.offset_;
			requires_grad_ = other.requires_grad_;
			grad_ = std::move(other.grad_);
			grad_fn_ = std::move(other.grad_fn_);
			other.offset_ = 0;
		}
		return *this;
	}

	//生成全零tensor
	Tensor Tensor::zeros(const std::vector<size_t>& shape) {
		return Tensor(shape);
	}

	//生成全一Tensor
	Tensor Tensor::ones(const std::vector<size_t>& shape) {
		Tensor t(shape);
		t.fill(1.0f);
		return t;
	}

	//所有元素填充指定值
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
	Tensor Tensor::rand(const std::vector<size_t>& shape, float low, float high) {
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

	//获取总元素数
	size_t Tensor::size() const {
		if (shape_.empty()) return 0;
		size_t total = 1;
		for (auto s : shape_) total *= s;
		return total;
	}

	// 是否为空
	bool Tensor::empty() const {
		return size() == 0;
	}

	//多维索引访问
	float& Tensor::at(const std::vector<size_t>& indices) {
		return (*data_)[offset_ + linear_index(indices)];
	}
	const float& Tensor::at(const std::vector<size_t>& indices) const {
		return (*data_)[offset_ + linear_index(indices)];
	}

	//一维线性索引访问
	float& Tensor::operator[](size_t index) {
		return (*data_)[offset_ + index];
	}
	const float& Tensor::operator[](size_t index)const {
		return (*data_)[offset_ + index];
	}

	// 获取底层数据指针，更高性能
	float* Tensor::data_ptr() {
		return data_->data() + offset_;
	}
	const float* Tensor::data_ptr() const {
		return data_->data() + offset_;
	}

	// 填充指定值
	void Tensor::fill(float value) {
		std::fill(data_->begin() + offset_, data_->begin() + offset_ + size(), value);
	}

	// 返回数据的深拷贝（不共享内存）
	Tensor Tensor::clone() const {
		Tensor t(shape_);
		for (size_t i = 0;i < size();i++) {
			(*t.data_)[i] = (*data_)[offset_ + i];
		}
		return t;
	}

	//重塑形状
	Tensor Tensor::reshape(const std::vector<size_t>& new_shape) const {
		size_t new_total = 1;
		for (auto s : new_shape) new_total *= s;
		if (new_total != size()) {
			throw std::invalid_argument(
				"Can not reshape tensor of size " + std::to_string(size()) +
				"to shape with " + std::to_string(new_total) + " elements"
			);
		}

		Tensor t = clone();
		t.shape_ = new_shape;
		t.compute_strides();
		return t;
	}

	//转置
	Tensor Tensor::transpose()const {
		if (ndim() != 2) {
			throw std::invalid_argument(
				"transpose() without args only works on 2D tensors"
			);
		}
		return transpose(0, 1);
	}

	// 通用转置（指定两个维度交换）
	Tensor Tensor::transpose(size_t dim0, size_t dim1) const {
		if (dim0 >= ndim() || dim1 >= ndim()) {
			throw std::out_of_range("Transpose dimensions out of range");
		}

		Tensor result(*this);
		std::swap(result.shape_[dim0], result.shape_[dim1]);
		std::swap(result.strides_[dim0], result.strides_[dim1]);
		return result;
	}

	// 去掉大小为 1 的维度
	Tensor Tensor::squeeze() const {
		Tensor result(*this);

		std::vector<size_t> new_shape;
		std::vector<size_t> new_strides;

		for (size_t i = 0;i < ndim();i++) {
			if (shape_[i] != 1) {
				new_shape.push_back(shape_[i]);
				new_strides.push_back(strides_[i]);
			}
		}

		result.shape_ = new_shape;
		result.strides_ = new_strides;
		return result;
	}

	// 去掉指定位置大小为 1 的维度
	Tensor Tensor::squeeze(size_t dim) const {
		if (dim >= ndim()) {
			throw std::out_of_range("Squeeze dimension out of range");
		}

		if (shape_[dim] != 1) {
			return *this;
		}

		Tensor result(*this);
		result.shape_.erase(result.shape_.begin() + dim);
		result.strides_.erase(result.strides_.begin() + dim);
		return result;
	}

	// 在指定位置插入大小为 1 的维度
	Tensor Tensor::unsqueeze(size_t dim) const {
		if (dim > ndim()) {
			throw std::out_of_range("Unsqueeze dimension out of range");
		}
		Tensor result(*this);

		result.shape_.insert(result.shape_.begin() + dim, 1);
		
		size_t stride;
		if (dim == result.strides_.size()) {
			stride = 1;
		}
		else {
			//为了连续性判断
			stride = result.strides_[dim] * result.shape_[dim + 1];
		}
		result.strides_.insert(result.strides_.begin() + dim, stride);

		return result;
	}

	//逐元素运算tensor tensor

	Tensor Tensor::operator+(const Tensor& other) const {
		Tensor result = elementwise_op(*this, other, [](float a, float b) {return a + b;});
		
		if (requires_grad_ || other.requires_grad_) {
			auto fn = std::make_shared<AddBackward>(*this, other);
			result.set_grad_fn(fn);
			result.set_requires_grad(true);
		}
		return result;
	}
	Tensor Tensor::operator-(const Tensor& other) const {
		Tensor result = elementwise_op(*this, other, [](float a, float b) {return a - b;});

		if (requires_grad_ || other.requires_grad_) {
			auto fn = std::make_shared<SubBackward>(*this, other);
			result.set_grad_fn(fn);
			result.set_requires_grad(true);
		}
		return result;
	}
	Tensor Tensor::operator*(const Tensor& other) const {
		Tensor result = elementwise_op(*this, other, [](float a, float b) {return a * b;});
		if (requires_grad_ || other.requires_grad_) {
			auto fn = std::make_shared<MulBackward>(*this, other);
			result.set_grad_fn(fn);
			result.set_requires_grad(true);
		}
		return result;
	}
	Tensor Tensor::operator/(const Tensor& other) const {
		Tensor result = elementwise_op(*this, other, [](float a, float b) {return a / b;});

		if (requires_grad_ || other.requires_grad_) {
			auto fn = std::make_shared<DivBackward>(*this, other);
			result.set_grad_fn(fn);
			result.set_requires_grad(true);
		}
		return result;
	}

	// 取负
	Tensor Tensor::operator-() const {
		Tensor result(shape_);
		for (size_t i = 0; i < size(); i++) {
			(*result.data_)[i] = -(*data_)[offset_ + i]; 
		}
		if (requires_grad_) {
			auto fn = std::make_shared<NegBackward>(*this);
			result.set_grad_fn(fn);
			result.set_requires_grad(true);
		}

		return result;
	}

	//逐元素运算: tensor&标量(标量在右侧)
	Tensor Tensor::operator+(float scalar) const {
		Tensor result(shape_);
		for (size_t i = 0;i < size();i++) {
			(*result.data_)[i] = (*data_)[offset_ + i]+scalar;
		}
		if (requires_grad_) {
			auto fn = std::make_shared<AddScalarBackward>(scalar, *this);
			result.set_grad_fn(fn);
			result.set_requires_grad(true);
		}
		return result;
	}
	Tensor Tensor::operator-(float scalar) const {
		Tensor result(shape_);
		for (size_t i = 0;i < size();i++) {
			(*result.data_)[i] = (*data_)[offset_ + i] - scalar;
		}
		if (requires_grad_) {
			auto fn = std::make_shared<LSubScalarBackward>(scalar, *this);
			result.set_grad_fn(fn);
			result.set_requires_grad(true);
		}
		return result;
	}
	Tensor Tensor::operator*(float scalar) const {
		Tensor result(shape_);
		for (size_t i = 0;i < size();i++) {
			(*result.data_)[i] = (*data_)[offset_ + i] * scalar;
		}
		if (requires_grad_) {
			auto fn = std::make_shared<MulScalarBackward>(scalar, *this);
			result.set_grad_fn(fn);
			result.set_requires_grad(true);
		}
		return result;
	}
	Tensor Tensor::operator/(float scalar) const {
		Tensor result(shape_);
		for (size_t i = 0;i < size();i++) {
			(*result.data_)[i] = (*data_)[offset_ + i] / scalar;
		}
		if (requires_grad_) {
			auto fn = std::make_shared<LDivScalarBackward>(scalar, *this);
			result.set_grad_fn(fn);
			result.set_requires_grad(true);
		}
		return result;
	}

	// 标量在左侧的运算（友元函数）
	Tensor operator+(float scalar, const Tensor& t) {
		return t + scalar;
	}
	Tensor operator-(float scalar, const Tensor& t) {
		Tensor result(t.shape_);
		for (size_t i = 0; i < t.size(); ++i) {
			(*result.data_)[i] = scalar - (*t.data_)[t.offset_ + i];
		}
		if (t.requires_grad()) {
			auto fn = std::make_shared<RSubScalarBackward>(scalar, t);
			result.set_grad_fn(fn);
			result.set_requires_grad(true);
		}
		return result;
	}
	Tensor operator*(float scalar, const Tensor& t) {
		return t * scalar;
	}
	Tensor operator/(float scalar, const Tensor& t) {
		Tensor result(t.shape_);
		for (size_t i = 0; i < t.size(); ++i) {
			(*result.data_)[i] = scalar / (*t.data_)[t.offset_ + i];
		}
		if (t.requires_grad()) {
			auto fn = std::make_shared<RDivScalarBackward>(scalar, t);
			result.set_grad_fn(fn);
			result.set_requires_grad(true);
		}
		return result;
	}

	// 矩阵乘法(参数Tensor在右边)
	Tensor Tensor::matmul(const Tensor& other) const {
		if (ndim() != 2 || other.ndim() != 2) {
			throw std::invalid_argument("matmul requires 2D tensors");
		}
		size_t M = shape_[0];
		size_t K = shape_[1];
		size_t N = other.shape_[1];
		if (K != other.shape_[0]) {
			throw std::invalid_argument(
				"matmul shape mismatch: (" + std::to_string(M) + "," +
				std::to_string(K) + ") x (" + std::to_string(other.shape_[0]) +
				"," + std::to_string(N) + ")"
			);
		}

		Tensor result({ M,N });
		for (size_t i = 0; i < M; ++i) {
			for (size_t j = 0; j < N; ++j) {
				float sum = 0.0f;
				for (size_t k = 0; k < K; ++k) {
					sum += at({ i, k }) * other.at({ k, j });
				}
				result.at({ i, j }) = sum;
			}
		}
		if (requires_grad_ || other.requires_grad_) {
			auto fn = std::make_shared<MatMulBackward>(*this, other);
			result.set_grad_fn(fn);
			result.set_requires_grad(true);
		}
		return result;
	}

	// 全局归约
	Tensor Tensor::sum() const {
		float s = 0.0f;
		for (size_t i = 0;i < size();i++) {
			s += (*data_)[offset_ + i];
		}
		Tensor result({ 1 }, { s });
		if (requires_grad_) {
			auto fn = std::make_shared<SumBackward>(*this);
			result.set_grad_fn(fn);
			result.set_requires_grad(true);
		}
		return result;
	}
	Tensor Tensor::mean() const {
		float s = sum().at({ 0 });
		return Tensor({ 1 }, { s / static_cast<float>(size()) });
	}
	Tensor Tensor::max() const {
		float m = (*data_)[offset_];
		for (size_t i = 0;i < size();i++) {
			m = std::max(m, (*data_)[offset_ + i]);
		}
		return Tensor({ 1 }, { m });
	}

	// 沿指定轴归约（keepdim = true 表示保留该维度）
	// axis是维度，沿着那个维度把值合并，那个维度就消失了
	Tensor Tensor::sum(size_t axis, bool keepdim) const {
		if (axis >= ndim()) {
			throw std::out_of_range("Sum axis out of range");
		}

		std::vector<size_t> out_shape;
		for (size_t i = 0;i < ndim();i++) {
			if (axis == i) {
				if (keepdim) {
					out_shape.push_back(1);
				}
			}
			else {
				out_shape.push_back(shape_[i]);
			}
		}
		//处理一维情况
		if (out_shape.empty()) {
			out_shape.push_back(1);
		}

		Tensor result(out_shape);

		std::vector<size_t> idx(ndim(), 0);
		for (size_t i = 0;i < size();i++) {
			std::vector<size_t> out_idx;
			for (size_t d = 0;d < ndim();d++) {
				if (d == axis) {
					if (keepdim) {
						out_idx.push_back(0);
					}
				}
				else {
					out_idx.push_back(idx[d]);
				}
			}

			result.at(out_idx) += at(idx);

			//遍历所有的位置
			for (int d = static_cast<int>(ndim()) - 1;d >= 0;d--) {
				idx[d]++;
				if (idx[d] < shape_[d]) break;
				idx[d] = 0;
			}
		}
		return result;
	}
	Tensor Tensor::mean(size_t axis, bool keepdim) const {
		Tensor s = sum(axis, keepdim);
		float count = static_cast<float>(shape_[axis]);
		return s / count;
	}
	Tensor Tensor::max(size_t axis, bool keepdim) const {
		//求每一维度axis的最大值
		if (axis >= ndim()) {
			throw std::out_of_range("Max axis out of range");
		}

		std::vector<size_t> out_shape;
		for (size_t i = 0;i < ndim();i++) {
			if (i == axis) {
				if (keepdim) {
					out_shape.push_back(1);
				}
			}
			else {
				out_shape.push_back(shape_[i]);
			}
		}

		if (out_shape.empty()) out_shape.push_back(1);

		//初始化为极小值
		Tensor result = Tensor::full(out_shape, -std::numeric_limits<float>::infinity());

		std::vector<size_t> idx(ndim(), 0);
		for (size_t i = 0;i < size();i++) {
			std::vector<size_t> out_idx;
			for (size_t d = 0;d < ndim();d++) {
				if (d == axis) {
					if (keepdim) {
						out_idx.push_back(0);
					}
				}
				else {
					out_idx.push_back(idx[d]);
				}
			}
			float& current = result.at(out_idx);
			current = std::max(current, at(idx));

			for (int d = static_cast<int>(ndim()) - 1;d >= 0;d--) {
				idx[d]++;
				if (idx[d] < shape_[d])break;
				idx[d] = 0;
			}
		}
		return result;

	}

	// 返回最大值的索引（沿指定轴）
	Tensor Tensor::argmax(size_t axis) const {
		if (axis >= ndim()) {
			throw std::out_of_range("Argmax axis out of range");
		}
		std::vector<size_t> out_shape;
		for (size_t i = 0;i < ndim();i++) {
			if (i != axis) out_shape.push_back(shape_[i]);
		}

		if (out_shape.empty()) out_shape.push_back(1);

		Tensor result(out_shape);
		Tensor max_vals = Tensor::full(out_shape, -std::numeric_limits<float>::infinity());

		std::vector<size_t> idx(ndim(), 0);
		for (size_t i = 0;i < size();i++) {
			std::vector<size_t> out_idx;
			for (size_t d = 0;d < ndim();d++) {
				if (d != axis)out_idx.push_back(idx[d]);
			}

			float val = at(idx);
			float& current_max = max_vals.at(out_idx);
			if (val > current_max) {
				current_max = val;
				result.at(out_idx) = static_cast<float>(idx[axis]);
			}

			for (int d = static_cast<int>(ndim()) - 1; d >= 0; --d) {
				idx[d]++;
				if (idx[d] < shape_[d]) break;
				idx[d] = 0;
			}
		}
		return result;
	}

	//数学函数
	Tensor Tensor::exp() const {
		Tensor result(shape_);
		for (size_t i = 0;i < size();i++) {
			(*result.data_)[i] = std::exp((*data_)[offset_ + i]);
		}
		if (requires_grad_) {
			auto fn = std::make_shared<ExpBackward>(*this, result);
			result.set_grad_fn(fn);
			result.set_requires_grad(true);
		}
		return result;
	}

	Tensor Tensor::log() const {
		Tensor result(shape_);
		for (size_t i = 0;i < size();i++) {
			(*result.data_)[i] = std::log((*data_)[offset_ + i]);
		}
		if (requires_grad_) {
			auto fn = std::make_shared<LogBackward>(*this);
			result.set_grad_fn(fn);
			result.set_requires_grad(true);
		}
		return result;
	}

	Tensor Tensor::pow(float exponent) const {
		Tensor result(shape_);
		for (size_t i = 0;i < size();i++) {
			(*result.data_)[i] = std::pow((*data_)[offset_ + i],exponent);
		}
		if (requires_grad_) {
			auto fn = std::make_shared<PowBackward>(exponent,*this);
			result.set_grad_fn(fn);
			result.set_requires_grad(true);
		}
		return result;
	}

	Tensor Tensor::sqrt() const {
		return pow(0.5f);
	}

	Tensor Tensor::abs() const {
		Tensor result(shape_);
		for (size_t i = 0;i < size();i++) {
			(*result.data_)[i] = std::abs((*data_)[offset_ + i]);
		}
		return result;
	}

	// 比较操作（返回 0/1 的 Tensor）

	Tensor Tensor::operator>(float scalar) const {
		Tensor result(shape_);
		for (size_t i = 0;i < size();i++) {
			(*result.data_)[i] = (*data_)[offset_ + i] > scalar ? 1.0f : 0.0f;
		}
		return result;
	}
	Tensor Tensor::operator<(float scalar) const {
		Tensor result(shape_);
		for (size_t i = 0;i < size();i++) {
			(*result.data_)[i] = (*data_)[offset_ + i] < scalar ? 1.0f : 0.0f;
		}
		return result;
	}
	Tensor Tensor::operator>=(float scalar) const {
		Tensor result(shape_);
		for (size_t i = 0;i < size();i++) {
			(*result.data_)[i] = (*data_)[offset_ + i] >= scalar ? 1.0f : 0.0f;
		}
		return result;
	}
	Tensor Tensor::operator<=(float scalar) const {
		Tensor result(shape_);
		for (size_t i = 0;i < size();i++) {
			(*result.data_)[i] = (*data_)[offset_ + i] <= scalar ? 1.0f : 0.0f;
		}
		return result;
	}
	Tensor Tensor::operator==(float scalar) const {
		Tensor result(shape_);
		for (size_t i = 0;i < size();i++) {
			(*result.data_)[i] = (*data_)[offset_ + i] == scalar ? 1.0f : 0.0f;
		}
		return result;
	}

	// 打印 Tensor 内容
	void Tensor::print(const std::string& name) const {
		if (!name.empty()) {
			std::cout << name << " ";
		}
		std::cout << *this << std::endl;
	}

	// 打印形状信息
	std::string Tensor::shape_str() const {
		std::ostringstream oss;
		oss << "(";
		for (size_t i = 0; i < shape_.size(); ++i) {
			oss << shape_[i];
			if (i + 1 < shape_.size()) oss << ", ";
		}
		oss << ")";
		return oss.str();
	}

	//流输出
	std::ostream& operator<<(std::ostream& os, const Tensor& t) {
		os << "Tensor(shape=" << t.shape_str() << ", data=[";
		size_t n = t.size();
		size_t max_show = 10; // 最多显示前 10 个元素
		for (size_t i = 0; i < std::min(n, max_show); ++i) {
			os << (*t.data_)[t.offset_ + i];
			if (i + 1 < std::min(n, max_show)) os << ", ";
		}
		if (n > max_show) {
			os << ", ... (" << n - max_show << " more)";
		}
		os << "])";
		return os;
	}

	// 获取底层数据引用
	const std::vector<float>& Tensor::data() const {
		return *data_;
	}

	// 反向传播相关
		//获取是否需要计算梯度
	bool Tensor::requires_grad() const {
		return requires_grad_;
	}

	// 设置是否需要计算梯度
	void Tensor::set_requires_grad(bool val) {
		requires_grad_ = val;
	}

	// 获取梯度 Tensor
	const std::shared_ptr<Tensor>& Tensor::grad() const {
		return *grad_;
	}

	//设置梯度 Tensor
	void Tensor::set_grad(const std::shared_ptr<Tensor>& g) {
		*grad_ = g;
	}

	// 获取生成该 Tensor 的 Autograd 函数
	const std::shared_ptr<Autograd>& Tensor::grad_fn() const {
		return grad_fn_;
	}

	// 设置生成该 Tensor 的 Autograd 函数
	void Tensor::set_grad_fn(const std::shared_ptr<Autograd>& fn) {
		grad_fn_ = fn;
	}

	void Tensor::backward() {
		set_grad(std::make_shared<Tensor>(Tensor::ones(shape_)));

		std::vector<Tensor*> topo;
		std::unordered_set<Tensor*> visited;

		std::function<void(Tensor*)> build_topo = [&](Tensor* t) {
			if (visited.count(t)) return;
			visited.insert(t);
			if (t->grad_fn_) {
				for (auto& inp : t->grad_fn_->inputs()) {
					build_topo(const_cast<Tensor*>(&inp));
				}
			}
			topo.push_back(t);
		};

		build_topo(this);
		std::reverse(topo.begin(), topo.end());

		for (auto* t : topo) {
			if (!t->grad_fn_ || !*t->grad_) continue;
			auto grads = t->grad_fn_->backward(**t->grad_);//得到对应位置的梯度
			auto& inps = t->grad_fn_->inputs();//得到创造出这个Tensor的Tensors
			for (size_t i = 0;i < inps.size();i++) {
				if (!inps[i].requires_grad_) continue;
				auto& inp = const_cast<Tensor&>(inps[i]);
				if (*inp.grad_) {
					inp.set_grad(std::make_shared<Tensor>(**inp.grad_ + grads[i]));
				}
				else {
					inp.set_grad(std::make_shared<Tensor>(grads[i]));
				}
			}
		}
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

	// 多维索引转一维线性索引
	size_t Tensor::linear_index(const std::vector<size_t>& indices) const {
		if (indices.size() != ndim()) {
			throw std::invalid_argument(
				"Index dimension mismatch: got" + std::to_string(indices.size()) +
				", expected " + std::to_string(ndim())
			);
		}
		size_t idx = 0;
		for (size_t i = 0;i < indices.size();i++) {
			if (indices[i] >= shape_[i]) {
				throw std::out_of_range(
					"Index " + std::to_string(indices[i]) +
					"out of range for dimension " + std::to_string(i) +
					"with size " + std::to_string(shape_[i])
				);
			}
			idx += indices[i] * strides_[i];
		}
		return idx;
	}

	// 广播两个 Tensor 的形状
	std::vector<size_t> Tensor::broadcast_shape(
		const std::vector<size_t>& a,
		const std::vector<size_t>& b
	) {
		size_t max_dim = std::max(a.size(), b.size());
		std::vector<size_t> result(max_dim);

		for (size_t i = 0; i < max_dim; ++i) {
			// 从右往左对齐
			size_t da = (i < a.size()) ? a[a.size() - 1 - i] : 1;
			size_t db = (i < b.size()) ? b[b.size() - 1 - i] : 1;

			if (da == db) {
				result[max_dim - 1 - i] = da;
			}
			else if (da == 1) {
				result[max_dim - 1 - i] = db;
			}
			else if (db == 1) {
				result[max_dim - 1 - i] = da;
			}
			else {
				throw std::invalid_argument(
					"Cannot broadcast shapes: dimension mismatch at axis " +
					std::to_string(max_dim - 1 - i) + " (" +
					std::to_string(da) + " vs " + std::to_string(db) + ")"
				);
			}
		}

		return result;
	}

	// 将索引映射到广播后的实际位置
	size_t Tensor::broadcast_index(
		const std::vector<size_t>& index,
		const std::vector<size_t>& shape
	) {
		size_t offset = index.size() - shape.size();
		size_t linear = 0;
		size_t stride = 1;
		for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
			size_t idx = (shape[i] == 1) ? 0 : index[i + offset];
			linear += idx * stride;
			stride *= shape[i];
		}
		return linear;
	}

	// 广播后执行逐元素运算
	Tensor Tensor::elementwise_op(
		const Tensor& a,
		const Tensor& b,
		std::function<float(float, float)> op) {
		auto result_shape = broadcast_shape(a.shape_, b.shape_);
		Tensor result(result_shape);
		size_t total = result.size();
		size_t ndim = result_shape.size();

		std::vector<size_t> idx(ndim, 0);
		for (size_t i = 0; i < total; ++i) {
			// 映射到 a 和 b 的索引
			size_t a_linear = broadcast_index(idx, a.shape_);
			size_t b_linear = broadcast_index(idx, b.shape_);

			(*result.data_)[i] = op(
				(*a.data_)[a.offset_ + a_linear],
				(*b.data_)[b.offset_ + b_linear]
			);

			// 递增索引
			for (int d = static_cast<int>(ndim) - 1; d >= 0; --d) {
				idx[d]++;
				if (idx[d] < result_shape[d]) break;
				idx[d] = 0;
			}
		}

		return result;
	}

}