#ifndef INC_D2P2_MATRIX_H_
#define INC_D2P2_MATRIX_H_
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
#include "d2p2.h"
#include <initializer_list>
#include <iostream>

namespace d2p2
{
class Dimensions
{
public:
    inline static constexpr uint32_t Max = 4;
    explicit Dimensions(uint32_t x0);
    Dimensions(uint32_t x0, uint32_t x1);
    Dimensions(uint32_t x0, uint32_t x1, uint32_t x2);
    Dimensions(uint32_t x0, uint32_t x1, uint32_t x2, uint32_t x3);

    Dimensions(const Dimensions& other);
    Dimensions& operator=(const Dimensions& other);

    uint32_t size() const
    {
        return size_;
    }
    uint32_t operator[](uint32_t index) const
    {
        return dimensions_[index];
    }

private:
    uint32_t size_;
    uint32_t dimensions_[Max];
};

/**
 * @brief
 */
class Tensor
{
public:
    inline static constexpr uint32_t Max = 4;

    Tensor();
    Tensor(uint32_t s0);
    Tensor(uint32_t s0, uint32_t s1);
    Tensor(uint32_t s0, uint32_t s1, uint32_t s2);
    Tensor(uint32_t s0, uint32_t s1, uint32_t s2, uint32_t s3);

    Tensor(uint32_t s0, std::initializer_list<float> args);
    Tensor(uint32_t s0, uint32_t s1, std::initializer_list<float> args);
    Tensor(uint32_t s0, uint32_t s1, uint32_t s2, std::initializer_list<float> args);
    Tensor(uint32_t s0, uint32_t s1, uint32_t s2, uint32_t s3, std::initializer_list<float> args);

    Tensor(uint32_t s0, const float* m);
    Tensor(uint32_t s0, uint32_t s1,const float* m);
    Tensor(uint32_t s0, uint32_t s1, uint32_t s2, const float* m);
    Tensor(uint32_t s0, uint32_t s1, uint32_t s2, uint32_t s3, const float* m);

    Tensor(const Tensor& other);
    Tensor(Tensor&& other) noexcept;
    ~Tensor();
    uint32_t dims() const{return dimensions_;}
    uint32_t size(uint32_t dim) const;
    void identity();

    const float& operator()(uint32_t i0) const;
    float& operator()(uint32_t i0);

    const float& operator()(uint32_t i0, uint32_t i1) const;
    float& operator()(uint32_t i0, uint32_t i1);

    const float& operator()(uint32_t i0, uint32_t i1, uint32_t i2) const;
    float& operator()(uint32_t i0, uint32_t i1, uint32_t i2);

    const float& operator()(uint32_t i0, uint32_t i1, uint32_t i2, uint32_t i3) const;
    float& operator()(uint32_t i0, uint32_t i1, uint32_t i2, uint32_t i3);

    Tensor& operator=(const Tensor& other);
    Tensor& operator=(Tensor&& other) noexcept;

    void setZeros();
    void setOnes();
    operator const float*() const;
    operator float*();
private:
    uint32_t sum_dims() const;
    uint32_t dimensions_;
    uint32_t size_[Max];
    float* m_;
};

//Tensor operator+(const Tensor& m0, const Tensor& m1);
//Tensor operator*(const Tensor& m0, const Tensor& m1);
//Tensor& operator*=(Tensor& m, float x);
//Tensor& operator/=(Tensor& m, float x);
//Tensor identity(const Tensor& m);
//Tensor step(const Tensor& m);
//Tensor sigmoid(const Tensor& m);
//Tensor relu(const Tensor& m);
//float max(const Tensor& m);
//float min(const Tensor& m);
//float sum(const Tensor& m);
//Tensor softmax(const Tensor& m);

std::ostream& operator<<(std::ostream& os, const Tensor& m);
} // namespace d2p2

#endif // INC_D2P2_MATRIX_H_
