#pragma once
#include <vector>
#include <string>
#include <memory>
#include <iostream>
#include <functional>
#include <initializer_list>
#include <random>
#include <numeric>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <stdexcept>
#include <sstream>

namespace nn {
	class Autograd;
	class Tensor
	{
	public:
		//默认构造
		Tensor();

		//从shape构造
		explicit Tensor(const std::vector<size_t>& shape);
		Tensor(const std::vector<size_t>& shape, const std::vector<float>& data);
		Tensor(const std::vector<size_t>& shape, const std::initializer_list<float>& data);
		//拷贝构造（浅拷贝，共享数据）
		Tensor(const Tensor& other);
		//移动构造
		Tensor(Tensor&& other)noexcept;
		//拷贝赋值
		Tensor& operator=(const Tensor& other);
		//移动赋值
		Tensor& operator=(Tensor&& other) noexcept;

		//静态工厂方法
		
		//全零tensor
		static Tensor zeros(const std::vector<size_t>& shape);

		//全一
		static Tensor ones(const std::vector<size_t>& shape);

		//指定位填充
		static Tensor full(const std::vector<size_t>& shape, float value);

		//标准正太分布随机初始化
		static Tensor randn(const std::vector<size_t>& shape);

		// 均匀分布随机初始化 [low, high)
		static Tensor rand(const std::vector<size_t>& shape, float low = 0.0f, float high = 1.0f);

		// 单位矩阵
		static Tensor eye(size_t n);

		//基本属性

		//形状
		const std::vector<size_t>& shape()const;

		//步长
		const std::vector<size_t>& strides() const;

		//维度
		size_t ndim()const;

		//获取总元素数
		size_t size() const;

		// 是否为空
		bool empty() const;

		//数据访问

		//多维索引访问
		float& at(const std::vector<size_t>& indices);
		const float& at(const std::vector<size_t>& indices) const;

		//一维线性索引访问
		float& operator[](size_t index);
		const float& operator[](size_t index)const;

		// 获取底层数据指针，更高性能
		float* data_ptr();
		const float* data_ptr() const;

		// 获取底层数据引用
		const std::vector<float>& data() const;

		// 填充指定值
		void fill(float value);

		// 返回数据的深拷贝（不共享内存）
		Tensor clone() const;

		//形状变换

		//重塑形状
		Tensor reshape(const std::vector<size_t>& new_shape) const;

		//转置
		Tensor transpose()const;

		// 通用转置（指定两个维度交换）
		Tensor transpose(size_t dim0, size_t dim1) const;

		// 去掉大小为 1 的维度
		Tensor squeeze() const;

		// 去掉指定位置大小为 1 的维度
		Tensor squeeze(size_t dim) const;

		// 在指定位置插入大小为 1 的维度
		Tensor unsqueeze(size_t dim) const;

		//逐元素运算tensor tensor

		Tensor operator+(const Tensor& other) const;
		Tensor operator-(const Tensor& other) const;
		Tensor operator*(const Tensor& other) const;  // element-wise 乘法
		Tensor operator/(const Tensor& other) const;

		// 取负
		Tensor operator-() const;

		//逐元素运算tensor 标量
		Tensor operator+(float scalar) const;
		Tensor operator-(float scalar) const;
		Tensor operator*(float scalar) const;
		Tensor operator/(float scalar) const;

		// 标量在左侧的运算（友元函数）
		friend Tensor operator+(float scalar, const Tensor& t);
		friend Tensor operator-(float scalar, const Tensor& t);
		friend Tensor operator*(float scalar, const Tensor& t);
		friend Tensor operator/(float scalar, const Tensor& t);

		// 矩阵乘法
		Tensor matmul(const Tensor& other) const;

		// 全局归约
		Tensor sum() const;
		Tensor mean() const;
		Tensor max() const;

		// 沿指定轴归约（keepdim 表示是否保留该维度）
		Tensor sum(size_t axis, bool keepdim = false) const;
		Tensor mean(size_t axis, bool keepdim = false) const;
		Tensor max(size_t axis, bool keepdim = false) const;

		// 返回最大值的索引（沿指定轴）
		Tensor argmax(size_t axis) const;

		//数学函数
		Tensor exp() const;
		Tensor log() const;
		Tensor pow(float exponent) const;
		Tensor sqrt() const;
		Tensor abs() const;

		// 比较操作（返回 0/1 的 Tensor）
		Tensor operator>(float scalar) const;
		Tensor operator<(float scalar) const;
		Tensor operator>=(float scalar) const;
		Tensor operator<=(float scalar) const;
		Tensor operator==(float scalar) const;

		// 打印 Tensor 内容
		void print(const std::string& name = "") const;

		// 打印形状信息
		std::string shape_str() const;

		// 流输出
		friend std::ostream& operator<<(std::ostream& os, const Tensor& t);

		// 反向传播相关
		//获取是否需要计算梯度
		bool requires_grad() const;

		// 设置是否需要计算梯度
		void set_requires_grad(bool val);

		// 获取梯度 Tensor
		const std::shared_ptr<Tensor>& grad() const;

		//设置梯度 Tensor
		void set_grad(const std::shared_ptr<Tensor>& g);

		// 获取生成该 Tensor 的 Autograd 函数
		const std::shared_ptr<Autograd>& grad_fn() const;

		// 设置生成该 Tensor 的 Autograd 函数
		void set_grad_fn(const std::shared_ptr<Autograd>& fn);

		void backward();


		~Tensor()=default;

	private:
		std::shared_ptr<std::vector<float>> data_;
		std::vector<size_t> shape_;
		std::vector<size_t> strides_;
		size_t offset_ = 0;

		bool requires_grad_ = false;
		std::shared_ptr<std::shared_ptr<Tensor>> grad_;
		std::shared_ptr<Autograd> grad_fn_;

		// 根据 shape 计算 strides
		void compute_strides();

		// 多维索引转一维线性索引
		size_t linear_index(const std::vector<size_t>& indices) const;

		// 广播两个 Tensor 的形状
		static std::vector<size_t> broadcast_shape(
			const std::vector<size_t>& a,
			const std::vector<size_t>& b
		);

		// 将索引映射到广播后的实际位置
		static size_t broadcast_index(
			const std::vector<size_t>& index,
			const std::vector<size_t>& shape
		);

		// 广播后执行逐元素运算
		static Tensor elementwise_op(
			const Tensor& a,
			const Tensor& b,
			std::function<float(float, float)> op
		);

	};
}
