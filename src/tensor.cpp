// clang-format off
/**
# License
This software is distributed under two licenses, choose whichever you like.

## MIT License
Copyright (c) 2023 Takuro Sakai

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Public Domain
This is free and unencumbered software released into the public domain.

Anyone is free to copy, modify, publish, use, compile, sell, or
distribute this software, either in source code form or as a compiled
binary, for any purpose, commercial or non-commercial, and by any
means.

In jurisdictions that recognize copyright laws, the author or authors
of this software dedicate any and all copyright interest in the
software to the public domain. We make this dedication for the benefit
of the public at large and to the detriment of our heirs and
successors. We intend this dedication to be an overt act of
relinquishment in perpetuity of all present and future rights to this
software under copyright law.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.

For more information, please refer to <http://unlicense.org>
*/
// clang-format on
#include "d2p2/tensor.h"
#include "src/ispc/matrix.ispc.h"
#include <algorithm>
#include <cstring>

namespace d2p2
{
Tensor::Tensor()
    : rows_(0)
    , cols_(0)
    , channels_(0)
    , m_(nullptr)
{
}

Tensor::Tensor(uint32_t rows, uint32_t cols, uint32_t channels)
    : rows_(rows)
    , cols_(cols)
    , channels_(channels)
    , m_(nullptr)
{
    assert(0 < rows_);
    assert(0 < cols_);
    assert(0 < channels_);
    m_ = static_cast<float*>(d2p2_malloc(sizeof(float) * rows_ * cols_ * channels_));
}

Tensor::Tensor(uint32_t rows, uint32_t cols, uint32_t channels, std::initializer_list<float> args)
    : rows_(rows)
    , cols_(cols)
    , channels_(channels)
    , m_(nullptr)
{
    assert(0 < rows_);
    assert(0 < cols_);
    assert(0 < channels_);
    assert((rows_ * cols_ * channels_) == args.size());
    m_ = static_cast<float*>(d2p2_malloc(sizeof(float) * rows_ * cols_ * channels_));
    for(uint32_t e = 0; e < channels_; ++e) {
        for(uint32_t r = 0; r < rows_; ++r) {
            for(uint32_t c = 0; c < cols_; ++c) {
                uint32_t i = (e * rows_ + r) * cols_ + c;
                m_[i] = args.begin()[i];
            }
        }
    }
}

Tensor::Tensor(uint32_t rows, uint32_t cols, uint32_t channels, float* m)
    : rows_(rows)
    , cols_(cols)
    , channels_(channels)
    , m_(m)
{
    assert(0 < rows_);
    assert(0 < cols_);
    assert(0 < channels_);
    assert(nullptr != m);
}

Tensor::Tensor(const Tensor& other)
    : rows_(other.rows_)
    , cols_(other.cols_)
    , channels_(other.channels_)
    , m_(nullptr)
{
    m_ = static_cast<float*>(d2p2_malloc(sizeof(float) * rows_ * cols_ * channels_));
    ::memcpy(m_, other.m_, sizeof(float) * rows_ * cols_ * channels_);
}

Tensor::Tensor(Tensor&& other) noexcept
    : rows_(other.rows_)
    , cols_(other.cols_)
    , channels_(other.channels_)
    , m_(other.m_)
{
    other.rows_ = 0;
    other.cols_ = 0;
    other.channels_ = 0;
    other.m_ = nullptr;
}

Tensor::~Tensor()
{
    d2p2_free(m_);
    rows_ = 0;
    cols_ = 0;
    channels_ = 0;
    m_ = nullptr;
}

uint32_t Tensor::rows() const
{
    return rows_;
}

uint32_t Tensor::cols() const
{
    return cols_;
}

uint32_t Tensor::channels() const
{
    return channels_;
}

void Tensor::identity()
{
    ::memset(m_, 0, sizeof(float) * rows_ * cols_ * channels_);
    uint32_t x = std::min(rows_, cols_);
    for(uint32_t i=0; i<channels_; ++i){
        for(uint32_t j=0; j<x; ++j){
            m_[(i*rows_ + j)*cols_ + j] = 1.0f;
        }
    }
}

const float& Tensor::operator()(uint32_t r, uint32_t c, uint32_t e) const
{
    assert(0 <= r && r < rows_);
    assert(0 <= c && c < cols_);
    assert(0 <= e && e < channels_);
    return m_[(e*rows_ + r)*cols_ + c];
}

float& Tensor::operator()(uint32_t r, uint32_t c, uint32_t e)
{
    assert(0 <= r && r < rows_);
    assert(0 <= c && c < cols_);
    assert(0 <= e && e < channels_);
    return m_[(e*rows_ + r)*cols_ + c];
}

const float& Tensor::operator()(uint32_t i) const
{
    assert(0 <= i && i < (rows_ * cols_ * channels_));
    return m_[i];
}

float& Tensor::operator()(uint32_t i)
{
    assert(0 <= i && i < (rows_ * cols_ * channels_));
    return m_[i];
}

Tensor& Tensor::operator=(const Tensor& other)
{
    if(this == &other) {
        return *this;
    }
    d2p2_free(m_);
    rows_ = other.rows_;
    cols_ = other.cols_;
    channels_ = other.channels_;
    m_ = static_cast<float*>(d2p2_malloc(sizeof(float) * rows_ * cols_ * channels_));
    ::memcpy(m_, other.m_, sizeof(float) * rows_ * cols_ * channels_);
    return *this;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept
{
    if(this == &other) {
        return *this;
    }
    d2p2_free(m_);
    rows_ = other.rows_;
    cols_ = other.cols_;
    channels_ = other.channels_;
    m_ = other.m_;
    other.rows_ = 0;
    other.cols_ = 0;
    other.channels_ = 0;
    other.m_ = nullptr;
    return *this;
}

Tensor::operator const float*() const
{
    return m_;
}

Tensor::operator float*()
{
    return m_;
}

const float& Tensor::get1d(uint32_t index, uint32_t e) const
{
    assert(index<rows_*cols_);
    assert(e<channels_);
    return m_[e*(rows_*cols_) + index];
}

float& Tensor::get1d(uint32_t index, uint32_t e)
{
    assert(index<rows_*cols_);
    assert(e<channels_);
    return m_[e*(rows_*cols_) + index];
}

const float& Tensor::get2d(uint32_t r, uint32_t c, uint32_t e) const
{
    return m_[channels_*(r*cols_ + c) + e];
}

float& Tensor::get2d(uint32_t r, uint32_t c, uint32_t e)
{
    return m_[channels_*(r*cols_ + c) + e];
}

void Tensor::setZeros()
{
    ::memset(m_, 0, rows_*cols_*channels_*sizeof(float));
}

void Tensor::setOnes()
{
    uint32_t size = rows_*cols_*channels_;
    for(uint32_t i=0; i<size; ++i){
        m_[i] = 1.0f;
    }
}

Tensor operator+(const Tensor& m0, const Tensor& m1)
{
    assert(m0.rows() == m1.rows());
    assert(m0.cols() == m1.cols());
    Tensor r(m0.rows(), m1.cols(), m1.channels());
    ispc::matrix_add(r, m0.rows() * m0.cols(), m0, m1);
    return r;
}

Tensor operator*(const Tensor& m0, const Tensor& m1)
{
    assert(m0.cols() == m1.rows());
    Tensor r(m0.rows(), m1.cols(), m1.channels());
    ispc::matrix_mul(r, m0.rows(), m0.cols(), m1.cols(), m0, m1);
    return r;
}

Tensor& operator*=(Tensor& m, float x)
{
    for(uint32_t i = 0; i < (m.rows() * m.cols() * m.channels()); ++i) {
        m(i) *= x;
    }
    return m;
}

Tensor& operator/=(Tensor& m, float x)
{
    float inv = 1.0f / x;
    for(uint32_t i = 0; i < (m.rows() * m.cols() * m.channels()); ++i) {
        m(i) *= inv;
    }
    return m;
}

Tensor identity(const Tensor& m)
{
    Tensor r(m.rows(), m.cols(), m.channels());
    ::memcpy(r, m, sizeof(float) * m.rows() * m.cols() * m.channels());
    return r;
}

Tensor step(const Tensor& m)
{
    Tensor r(m.rows(), m.cols(), m.channels());
    ispc::step(m.rows() * m.cols() * m.channels(), r, m);
    return r;
}

Tensor sigmoid(const Tensor& m)
{
    Tensor r(m.rows(), m.cols(), m.channels());
    ispc::sigmoid(m.rows() * m.cols() * m.channels(), r, m);
    return r;
}

Tensor relu(const Tensor& m)
{
    Tensor r(m.rows(), m.cols(), m.channels());
    ispc::relu(m.rows() * m.cols() * m.channels(), r, m);
    return r;
}

float max(const Tensor& m)
{
    assert(0 < m.rows());
    assert(0 < m.cols());
    assert(0 < m.channels());
    return ispc::max_p(m.rows() * m.cols() * m.channels(), m);
}

float min(const Tensor& m)
{
    assert(0 < m.rows());
    assert(0 < m.cols());
    assert(0 < m.channels());
    return ispc::min_p(m.rows() * m.cols() * m.channels(), m);
}

float sum(const Tensor& m)
{
    return ispc::sum(m.rows() * m.cols() * m.channels(), m);
}

Tensor softmax(const Tensor& m)
{
    Tensor r(m.rows(), m.cols(), m.channels());
    ispc::softmax(m.rows() * m.cols() * m.channels(), r, m);
    return r;
}

std::ostream& operator<<(std::ostream& os, const Tensor& m)
{
    os << '[';
    for(uint32_t r = 0; r < m.rows(); ++r) {
        os << '[';
        for(uint32_t c = 0; c < m.cols(); ++c) {
            os << '[';
            for(uint32_t e = 0; e < m.channels(); ++e) {
                os << m(r, c, e) << ' ';
            }
            os << ']';
        }
        os << ']';
    }
    os << ']';
    return os;
}

std::ostream& print_numpy(std::ostream& os, const Tensor& m)
{
    os << '[';
    for(uint32_t e = 0; e < m.channels(); ++e) {
        os << '[';
        for(uint32_t r = 0; r < m.rows(); ++r) {
            os << '[';
            for(uint32_t c = 0; c < m.cols(); ++c) {
                os << m(r, c, e) << ' ';
            }
            os << ']';
        }
        os << ']';
    }
    os << ']';
    return os;
}
} // namespace d2p2
