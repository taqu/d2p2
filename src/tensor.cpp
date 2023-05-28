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
#include <cstring>
#include <cstdarg>
#include <algorithm>

namespace d2p2
{
//--------------------------------------------------------
Dimensions::Dimensions(uint32_t x0)
    :size_(1)
    ,dimensions_{}
{
    dimensions_[0] = x0;
}

Dimensions::Dimensions(uint32_t x0, uint32_t x1)
    :size_(2)
    ,dimensions_{}
{
    dimensions_[0] = x0;
    dimensions_[1] = x1;
}

Dimensions::Dimensions(uint32_t x0, uint32_t x1, uint32_t x2)
:size_(3)
    ,dimensions_{}
{
    dimensions_[0] = x0;
    dimensions_[1] = x1;
    dimensions_[2] = x2;
}

Dimensions::Dimensions(uint32_t x0, uint32_t x1, uint32_t x2, uint32_t x3)
:size_(4)
    ,dimensions_{}
{
    dimensions_[0] = x0;
    dimensions_[1] = x1;
    dimensions_[2] = x2;
    dimensions_[3] = x3;
}

Dimensions::Dimensions(const Dimensions& other)
    :size_(other.size_)
{
    for(uint32_t i=0; i<Max; ++i){
        dimensions_[i] = other.dimensions_[i];
    }
}

Dimensions& Dimensions::operator=(const Dimensions& other)
{
    if(this == &other){
        return *this;
    }
    size_ = other.size_;
    for(uint32_t i=0; i<Max; ++i){
        dimensions_[i] = other.dimensions_[i];
    }
    return *this;
}

//--------------------------------------------------------
Tensor::Tensor()
    : dimensions_(0)
    , size_{}
    , m_(nullptr)
{
}

Tensor::Tensor(uint32_t s0)
    : dimensions_(1)
    , size_{}
    , m_(nullptr)
{
    m_ = static_cast<float*>(d2p2_malloc(sizeof(float) * s0));
    size_[0] = s0;
}

Tensor::Tensor(uint32_t s0, uint32_t s1)
    : dimensions_(2)
    , size_{}
    , m_(nullptr)
{
    m_ = static_cast<float*>(d2p2_malloc(sizeof(float) * (s0*s1)));
    size_[0] = s0;
    size_[1] = s1;
}

Tensor::Tensor(uint32_t s0, uint32_t s1, uint32_t s2)
    : dimensions_(3)
    , size_{}
    , m_(nullptr)
{
    m_ = static_cast<float*>(d2p2_malloc(sizeof(float) * (s0*s1*s2)));
    size_[0] = s0;
    size_[1] = s1;
    size_[2] = s2;
}

Tensor::Tensor(uint32_t s0, uint32_t s1, uint32_t s2, uint32_t s3)
    : dimensions_(4)
    , size_{}
    , m_(nullptr)
{
    m_ = static_cast<float*>(d2p2_malloc(sizeof(float) * (s0*s1*s2*s3)));
    size_[0] = s0;
    size_[1] = s1;
    size_[2] = s2;
    size_[3] = s3;
}

Tensor::Tensor(uint32_t s0, std::initializer_list<float> args)
    : dimensions_(1)
    , size_{}
    , m_(nullptr)
{
    m_ = static_cast<float*>(d2p2_malloc(sizeof(float) * s0));
    size_[0] = s0;
    assert(size_[0] == args.size());
    uint32_t count = 0;
    for(const float x : args){
        m_[count] = x;
        ++count;
    }
}

Tensor::Tensor(uint32_t s0, uint32_t s1, std::initializer_list<float> args)
    : dimensions_(2)
    , size_{}
    , m_(nullptr)
{
    m_ = static_cast<float*>(d2p2_malloc(sizeof(float) * (s0*s1)));
    size_[0] = s0;
    size_[1] = s1;
    const float* src = args.begin();
    for(uint32_t i = 0; i < size_[0]; ++i) {
        uint32_t r0 = i*size_[1];
        for(uint32_t j = 0; j < size_[1]; ++j) {
            uint32_t index = r0 + j;
            m_[index] = src[index];
        }
    }
}

Tensor::Tensor(uint32_t s0, uint32_t s1, uint32_t s2, std::initializer_list<float> args)
    : dimensions_(3)
    , size_{}
    , m_(nullptr)
{
    m_ = static_cast<float*>(d2p2_malloc(sizeof(float) * (s0*s1*s2)));
    size_[0] = s0;
    size_[1] = s1;
    size_[2] = s2;
    const float* src = args.begin();
    for(uint32_t i = 0; i < size_[0]; ++i) {
        uint32_t r0 = i*size_[1]*size_[2];
        for(uint32_t j = 0; j < size_[1]; ++j) {
            uint32_t r1 = j*size_[2];
            for(uint32_t k = 0; k < size_[2]; ++k) {
                uint32_t index = r0 + r1 + k;
                m_[index] = src[index];
            }
        }
    }
}

Tensor::Tensor(uint32_t s0, uint32_t s1, uint32_t s2, uint32_t s3, std::initializer_list<float> args)
    : dimensions_(4)
    , size_{}
    , m_(nullptr)
{
    m_ = static_cast<float*>(d2p2_malloc(sizeof(float) * (s0*s1*s2*s3)));
    size_[0] = s0;
    size_[1] = s1;
    size_[2] = s2;
    size_[3] = s3;
    const float* src = args.begin();
    for(uint32_t i = 0; i < size_[0]; ++i) {
        uint32_t r0 = i * size_[1] * size_[2] * size_[3];
        for(uint32_t j = 0; j < size_[1]; ++j) {
            uint32_t r1 = j * size_[2] * size_[3];
            for(uint32_t k = 0; k < size_[2]; ++k) {
                uint32_t r2 = k * size_[3];
                for(uint32_t l = 0; l < size_[3]; ++l) {
                    uint32_t index = r0 + r1 + r2 + l;
                    m_[index] = src[index];
                }
            }
        }
    }
}

Tensor::Tensor(uint32_t s0, const float* m)
    : dimensions_(1)
    , size_{}
    , m_(nullptr)
{
    m_ = static_cast<float*>(d2p2_malloc(sizeof(float) * s0));
    size_[0] = s0;
    ::memcpy(m_, m, sizeof(float)*s0);
}

Tensor::Tensor(uint32_t s0, uint32_t s1, const float* m)
    : dimensions_(2)
    , size_{}
    , m_(nullptr)
{
    m_ = static_cast<float*>(d2p2_malloc(sizeof(float) * (s0*s1)));
    size_[0] = s0;
    size_[1] = s1;
    ::memcpy(m_, m, sizeof(float)*s0*s1);
}

Tensor::Tensor(uint32_t s0, uint32_t s1, uint32_t s2, const float* m)
    : dimensions_(3)
    , size_{}
    , m_(nullptr)
{
    m_ = static_cast<float*>(d2p2_malloc(sizeof(float) * (s0*s1*s2)));
    size_[0] = s0;
    size_[1] = s1;
    size_[2] = s2;
    ::memcpy(m_, m, sizeof(float)*s0*s1*s2);
}

Tensor::Tensor(uint32_t s0, uint32_t s1, uint32_t s2, uint32_t s3, const float* m)
    : dimensions_(4)
    , size_{}
    , m_(nullptr)
{
    m_ = static_cast<float*>(d2p2_malloc(sizeof(float) * (s0*s1*s2*s3)));
    size_[0] = s0;
    size_[1] = s1;
    size_[2] = s2;
    size_[3] = s3;
    ::memcpy(m_, m, sizeof(float)*s0*s1*s2*s3);
}

Tensor::Tensor(const Tensor& other)
    : dimensions_(other.dimensions_)
    , size_{}
    , m_(nullptr)
{
    ::memcpy(size_, other.size_, sizeof(uint32_t)*Max);
    uint32_t total = sum_dims();
    m_ = static_cast<float*>(d2p2_malloc(sizeof(float) * total));
    ::memcpy(m_, other.m_, sizeof(float) * total);
}

Tensor::Tensor(Tensor&& other) noexcept
    : dimensions_(other.dimensions_)
    , size_{}
    , m_(other.m_)
{
    ::memcpy(size_, other.size_, sizeof(uint32_t)*Max);
    other.dimensions_ = 0;
    ::memset(other.size_, 0, sizeof(uint32_t)*Max);
    other.m_ = nullptr;
}

Tensor::~Tensor()
{
    d2p2_free(m_);
    dimensions_ = 0;
    ::memset(size_, 0, sizeof(uint32_t)*Max);
    m_ = nullptr;
}

uint32_t Tensor::size(uint32_t dim) const
{
    assert(dim<Max);
    return size_[dim];
}

void Tensor::identity()
{
    uint32_t total = sum_dims();
    ::memset(m_, 0, sizeof(float) * total);
    uint32_t x = size_[0];
    for(uint32_t i=1; i<dimensions_; ++i){
        x = std::min(x, size_[i]);
    }
    for(uint32_t i=0; i<x; ++i){
        uint32_t index = i;
        for(uint32_t j=1; j<dimensions_; ++j){
            index *= size_[j];
            index += i;
        }
        m_[index] = 1.0f;
    }
}

const float& Tensor::operator()(uint32_t i0) const
{
    assert(1 == dimensions_);
    assert(i0 < size_[0]);
    return m_[i0];
}

float& Tensor::operator()(uint32_t i0)
{
    assert(1 == dimensions_);
    assert(i0 < size_[0]);
    return m_[i0];
}

const float& Tensor::operator()(uint32_t i0, uint32_t i1) const
{
    assert(2 == dimensions_);
    assert(i0 < size_[0]);
    assert(i1 < size_[1]);
    uint32_t index = i0*size_[1] + i1;
    return m_[index];
}

float& Tensor::operator()(uint32_t i0, uint32_t i1)
{
    assert(2 == dimensions_);
    assert(i0 < size_[0]);
    assert(i1 < size_[1]);
    uint32_t index = i0*size_[1] + i1;
    return m_[index];
}

const float& Tensor::operator()(uint32_t i0, uint32_t i1, uint32_t i2) const
{
    assert(3 == dimensions_);
    assert(i0 < size_[0]);
    assert(i1 < size_[1]);
    assert(i2 < size_[2]);
    uint32_t index = (i0*size_[1] + i1)*size_[2] + i2;
    return m_[index];
}

float& Tensor::operator()(uint32_t i0, uint32_t i1, uint32_t i2)
{
    assert(3 == dimensions_);
    assert(i0 < size_[0]);
    assert(i1 < size_[1]);
    assert(i2 < size_[2]);
    uint32_t index = (i0*size_[1] + i1)*size_[2] + i2;
    return m_[index];
}

const float& Tensor::operator()(uint32_t i0, uint32_t i1, uint32_t i2, uint32_t i3) const
{
    assert(4 == dimensions_);
    assert(i0 < size_[0]);
    assert(i1 < size_[1]);
    assert(i2 < size_[2]);
    assert(i3 < size_[3]);
    uint32_t index = ((i0*size_[1] + i1)*size_[2] + i2)*size_[3] + i3;
    return m_[index];
}

float& Tensor::operator()(uint32_t i0, uint32_t i1, uint32_t i2, uint32_t i3)
{
    assert(4 == dimensions_);
    assert(i0 < size_[0]);
    assert(i1 < size_[1]);
    assert(i2 < size_[2]);
    assert(i3 < size_[3]);
    uint32_t index = ((i0*size_[1] + i1)*size_[2] + i2)*size_[3] + i3;
    return m_[index];
}

Tensor& Tensor::operator=(const Tensor& other)
{
    if(this == &other) {
        return *this;
    }
    d2p2_free(m_);
    dimensions_ = other.dimensions_;
    ::memcpy(size_, other.size_, sizeof(uint32_t)*Max);
    uint32_t total = sum_dims();
    m_ = static_cast<float*>(d2p2_malloc(sizeof(float) * total));
    ::memcpy(m_, other.m_, sizeof(float) * total);
    return *this;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept
{
    if(this == &other) {
        return *this;
    }
    d2p2_free(m_);
    dimensions_ = other.dimensions_;
    ::memcpy(size_, other.size_, sizeof(uint32_t)*Max);
    m_ = other.m_;
    other.dimensions_ = 0;
    ::memset(other.size_, 0, sizeof(uint32_t)*Max);
    other.m_ = nullptr;
    return *this;
}

void Tensor::setZeros()
{
    uint32_t total = sum_dims();
    ::memset(m_, 0, sizeof(float)*total);
}

void Tensor::setOnes()
{
    uint32_t total = sum_dims();
    for(uint32_t i=0; i<total; ++i){
        m_[i] = 1.0f;
    }
}

Tensor::operator const float*() const
{
    return m_;
}

    Tensor::operator float*()
{
        return m_;
        }

uint32_t Tensor::sum_dims() const
{
    uint32_t total = 1;
    for(uint32_t i=0; i<dimensions_; ++i){
        total *= size_[i];
    }
    return total;
}

#if 0
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
#endif

namespace
{
    void print1(std::ostream& os, const Tensor& m)
    {
        os << '[';
        for(uint32_t i0 = 0; i0 < m.size(0); ++i0) {
            os << m(i0) << ' ';
        }
        os << ']';
    }

    void print2(std::ostream& os, const Tensor& m)
    {
        os << '[';
        for(uint32_t i0 = 0; i0 < m.size(0); ++i0) {
            os << '[';
            for(uint32_t i1 = 0; i1 < m.size(1); ++i1) {
                os << m(i0, i1) << ' ';
            }
            os << ']';
        }
        os << ']';
    }

    void print3(std::ostream& os, const Tensor& m)
    {
        os << '[';
        for(uint32_t i0 = 0; i0 < m.size(0); ++i0) {
            os << '[';
            for(uint32_t i1 = 0; i1 < m.size(1); ++i1) {
                os << '[';
                for(uint32_t i2 = 0; i2 < m.size(2); ++i2) {
                    os << m(i0,i1,i2) << ' ';
                }
                os << ']';
            }
            os << ']';
        }
        os << ']';
    }

    void print4(std::ostream& os, const Tensor& m)
    {
        os << '[';
        for(uint32_t i0 = 0; i0 < m.size(0); ++i0) {
            os << '[';
            for(uint32_t i1 = 0; i1 < m.size(1); ++i1) {
                os << '[';
                for(uint32_t i2 = 0; i2 < m.size(2); ++i2) {
                    os << '[';
                    for(uint32_t i3 = 0; i3 < m.size(3); ++i3) {
                    os << m(i0,i1,i2,i3) << ' ';
                    }
                    os << ']';
                }
                os << ']';
            }
            os << ']';
        }
        os << ']';
    }
} // namespace

std::ostream& operator<<(std::ostream& os, const Tensor& m)
{
    switch(m.dims()){
    case 1:
        print1(os, m);
        break;
    case 2:
        print2(os, m);
        break;
    case 3:
        print3(os, m);
        break;
    case 4:
        print4(os, m);
        break;
    default:
        assert(false);
        break;
    }
    return os;
}
} // namespace d2p2

