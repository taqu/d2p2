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
/**
 * @brief
 */
class Tensor
{
public:
    Tensor();
    Tensor(uint32_t rows, uint32_t cols, uint32_t channels);
    Tensor(uint32_t rows, uint32_t cols, uint32_t channels, std::initializer_list<float> args);
    Tensor(uint32_t rows, uint32_t cols, uint32_t channels, float* m);
    Tensor(const Tensor& other);
    Tensor(Tensor&& other) noexcept;
    ~Tensor();
    uint32_t rows() const;
    uint32_t cols() const;
    uint32_t channels() const;
    void identity();

    const float& operator()(uint32_t r, uint32_t c, uint32_t e) const;
    float& operator()(uint32_t r, uint32_t c, uint32_t e);
    const float& operator()(uint32_t i) const;
    float& operator()(uint32_t i);

    Tensor& operator=(const Tensor& other);
    Tensor& operator=(Tensor&& other) noexcept;

    operator const float*() const;
    operator float*();

    const float& get1d(uint32_t index, uint32_t e) const;
    float& get1d(uint32_t index, uint32_t e);
    const float& get2d(uint32_t r, uint32_t c, uint32_t e) const;
    float& get2d(uint32_t r, uint32_t c, uint32_t e);

    void setZeros();
    void setOnes();
private:
    uint32_t rows_;
    uint32_t cols_;
    uint32_t channels_;
    float* m_;
};

Tensor operator+(const Tensor& m0, const Tensor& m1);
Tensor operator*(const Tensor& m0, const Tensor& m1);
Tensor& operator*=(Tensor& m, float x);
Tensor& operator/=(Tensor& m, float x);
Tensor identity(const Tensor& m);
Tensor step(const Tensor& m);
Tensor sigmoid(const Tensor& m);
Tensor relu(const Tensor& m);
float max(const Tensor& m);
float min(const Tensor& m);
float sum(const Tensor& m);
Tensor softmax(const Tensor& m);

std::ostream& operator<<(std::ostream& os, const Tensor& m);
std::ostream& print_numpy(std::ostream& os, const Tensor& m);
} // namespace d2p2

#endif // INC_D2P2_MATRIX_H_
