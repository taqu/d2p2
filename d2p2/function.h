#ifndef INC_D2P2_FUNCTION_H_
#define INC_D2P2_FUNCTION_H_
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
#include "tensor.h"
#include <tuple>

namespace d2p2
{
//---------------------------------------------
/**
 * @brief
 */
class IFunction
{
public:
    virtual ~IFunction() noexcept;

protected:
    IFunction();
};

//---------------------------------------------
/**
 * @brief
 */
struct Conv
{
public:
    inline static constexpr uint32_t Invalid = 0xFFFF'FFFFUL;

    enum class PaddingMode
    {
        Zeros,
        Reflect,
        Replicate,
        Repeat,
    };

    static std::tuple<int32_t, int32_t> kernel_range(uint32_t kernel_size);
    static std::tuple<uint32_t, uint32_t> conv_range(uint32_t kernel_size, uint32_t padding, uint32_t size);
    static uint32_t transpose_offset(uint32_t kernel_size);

    static uint32_t sample_zeros(uint32_t p, int32_t offset, uint32_t size);
    static uint32_t sample_reflect(uint32_t p, int32_t offset, uint32_t size);
    static uint32_t sample_replicate(uint32_t p, int32_t offset, uint32_t size);
    static uint32_t sample_repeat(uint32_t p, int32_t offset, uint32_t size);

private:
    Conv() = delete;
    ~Conv() = delete;
    Conv(const Conv&) = delete;
    Conv& operator=(const Conv&) = delete;
};

//---------------------------------------------
/**
 * @brief
 */
class Linear: public IFunction
{
public:
    Linear();
    Linear(Linear&& other) noexcept;
    Linear(uint32_t input_features, uint32_t output_features);
    virtual ~Linear() noexcept;

    Tensor operator()(const Tensor& tensor) const;
    const Tensor& weights() const;
    void weights(std::initializer_list<float> args);
    const Tensor& bias() const;
    void bias(std::initializer_list<float> args);

private:
    Linear(const Linear&) = delete;
    Linear& operator=(const Linear&) = delete;

    uint32_t input_features_;
    uint32_t output_features_;
    Tensor weights_;
    Tensor bias_;
};

//---------------------------------------------
/**
 * @brief
 */
class Conv1d: public IFunction
{
public:
    Conv1d();
    Conv1d(Conv1d&& other) noexcept;
    Conv1d(uint32_t input_channels, uint32_t output_channels, uint32_t kernel_size, uint32_t stride = 1, uint32_t padding = 0, Conv::PaddingMode padding_mode = Conv::PaddingMode::Zeros);
    virtual ~Conv1d() noexcept;

    Tensor operator()(const Tensor& tensor) const;

    const Tensor& weights() const;
    void weights(std::initializer_list<float> args);
    const Tensor& bias() const;
    void bias(std::initializer_list<float> args);

private:
    Conv1d(const Conv1d&) = delete;
    Conv1d& operator=(const Conv1d&) = delete;

    uint32_t input_channels_;
    uint32_t output_channels_;
    uint32_t kernel_size_;
    uint32_t stride_;
    uint32_t padding_;
    Conv::PaddingMode padding_mode_;
    Tensor weights_;
    Tensor bias_;
};

//---------------------------------------------
/**
 * @brief
 */
class ConvTranspose1d: public IFunction
{
public:
    ConvTranspose1d();
    ConvTranspose1d(ConvTranspose1d&& other) noexcept;
    ConvTranspose1d(uint32_t input_channels, uint32_t output_channels, uint32_t kernel_size, uint32_t stride = 1, uint32_t padding = 0, Conv::PaddingMode padding_mode = Conv::PaddingMode::Zeros);
    virtual ~ConvTranspose1d() noexcept;

    Tensor operator()(const Tensor& tensor) const;
    const Tensor& weights() const;
    void weights(std::initializer_list<float> args);
    const Tensor& bias() const;
    void bias(std::initializer_list<float> args);

private:
    ConvTranspose1d(const ConvTranspose1d&) = delete;
    ConvTranspose1d& operator=(const ConvTranspose1d&) = delete;

    uint32_t input_channels_;
    uint32_t output_channels_;
    uint32_t kernel_size_;
    uint32_t stride_;
    uint32_t padding_;
    Conv::PaddingMode padding_mode_;
    Tensor weights_;
    Tensor bias_;
};

//---------------------------------------------
/**
 * @brief
 */
class Conv2d: public IFunction
{
public:
    Conv2d();
    Conv2d(Conv2d&& other) noexcept;
    Conv2d(uint32_t input_channels, uint32_t output_channels, uint32_t kernel_size, uint32_t stride = 1, uint32_t padding = 0, Conv::PaddingMode padding_mode = Conv::PaddingMode::Zeros);
    virtual ~Conv2d() noexcept;

    Tensor operator()(const Tensor& tensor) const;
    const Tensor& weights() const;
    void weights(std::initializer_list<float> args);
    const Tensor& bias() const;
    void bias(std::initializer_list<float> args);

private:
    Conv2d(const Conv2d&) = delete;
    Conv2d& operator=(const Conv2d&) = delete;

    uint32_t input_channels_;
    uint32_t output_channels_;
    uint32_t kernel_size_;
    uint32_t stride_;
    uint32_t padding_;
    Conv::PaddingMode padding_mode_;
    Tensor weights_;
    Tensor bias_;
};

template<class T>
void mul(uint32_t rows, uint32_t cols, uint32_t dcols, T* dst, const T* m0, const T* m1)
{
    for(uint32_t sr = 0; sr < rows; ++sr) {
        for(uint32_t dc = 0; dc < dcols; ++dc) {
            T t = 0;
            for(uint32_t sc = 0; sc < cols; ++sc) {
                t += m0[sr * cols + sc] * m1[sc * dcols + dc];
            }
            dst[sr * cols + dc] = t;
        }
    }
}

} // namespace d2p2
#endif // INC_D2P2_FUNCTION_H_
