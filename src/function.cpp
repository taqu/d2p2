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
#include "d2p2/function.h"
#include "ispc/matrix.ispc.h"
#include <functional>
#include <utility>

namespace d2p2
{
namespace
{
    uint32_t src2dst_size(uint32_t x, uint32_t kernel_size, uint32_t stride, uint32_t padding)
    {
        int32_t ix = (static_cast<int32_t>(x + 2*padding) - (kernel_size - 1) - 1) / stride + 1;
        return 0 < ix ? static_cast<uint32_t>(ix) : 1;
    }

    uint32_t src2dst_index(uint32_t x, uint32_t kernel_size, uint32_t stride, uint32_t /*padding*/)
    {
        assert(0<kernel_size);
        assert(0<stride);
        uint32_t half = kernel_size/2;
        uint32_t offset = 0;
        if(0 == (kernel_size&0x01U)){
            assert(0<half);
            offset = (half-1);
        }else{
            offset = half;
        }
        assert(offset<=x);
        return (x-offset)/stride;
    }

    uint32_t transpose_src2dst_size(uint32_t size, uint32_t kernel_size, uint32_t stride, uint32_t padding)
    {
        assert(0<size);
        assert(0<kernel_size);
        uint32_t padd = (padding+1)<=kernel_size? (kernel_size-1-padding) : 0;
        uint32_t r;
        if(0 == (0x01U & size)) {
            r = (size - 1) * stride - 2 * padding + (kernel_size - 1);
        } else {
            r = size * stride - 2 * padding + (kernel_size - 1);
        }
        return r;
    }

} // namespace

//--- IFunction
//-------------------------------------------------------------------
IFunction::IFunction()
{
}

IFunction::~IFunction() noexcept
{
}

//--- Conv
//-------------------------------------------------------------------
std::tuple<int32_t, int32_t> Conv::kernel_range(uint32_t kernel_size, uint32_t padding, uint32_t size)
{
    assert(kernel_size<=size);
    int32_t left, right;
    uint32_t half = kernel_size / 2;
    if(0 == (0x01U & kernel_size)) {
        left = (0 < half) ? -static_cast<int32_t>(half - 1) : 0;
        right = static_cast<int32_t>(half);
    } else {
        left = -static_cast<int32_t>(half);
        right = static_cast<int32_t>(half);
    }
    return {left, right};
}

std::tuple<uint32_t, uint32_t> Conv::conv_range(uint32_t kernel_size, uint32_t padding, uint32_t size)
{
    assert(0<kernel_size);
    assert(0<size);
    assert(kernel_size<=size);
    uint32_t start, end;
    uint32_t half = kernel_size / 2;
    if(0 == (0x01U & kernel_size)) {
        start = (0<half)?half - 1 : 0;
        end = size-half-1;
    } else {
        start = half;
        end = size-half-1;
    }
    if(padding<start){
        start -= padding;
    }
    if(end<=(size-1)){
        end += padding;
    }
    return {start, end};
}

uint32_t Conv::sample_zeros(uint32_t p, int32_t offset, uint32_t size)
{
    int32_t ip = static_cast<int32_t>(p) + offset;
    int32_t isize = static_cast<int32_t>(size);
    if(0 <= ip && ip < isize) {
        return static_cast<uint32_t>(ip);
    }
    if(ip < 0) {
        return Invalid;
    }
    if(isize <= ip) {
        return Invalid;
    }
    return static_cast<uint32_t>(ip);
}

uint32_t Conv::sample_reflect(uint32_t p, int32_t offset, uint32_t size)
{
    int32_t ip = static_cast<int32_t>(p) + offset;
    if(ip < 0) {
        return static_cast<uint32_t>(-ip);
    }
    uint32_t up = static_cast<uint32_t>(ip);
    if(size <= up) {
        uint32_t dp = up - size + 1;
        assert(dp < size);
        return p - dp;
    }
    return up;
}

uint32_t Conv::sample_replicate(uint32_t p, int32_t offset, uint32_t size)
{
    int32_t ip = static_cast<int32_t>(p) + offset;
    uint32_t up;
    if(ip < 0) {
        up = 0;
    } else if(size <= static_cast<uint32_t>(up)) {
        assert(0 < size);
        up = size - 1;
    } else {
        up = static_cast<uint32_t>(ip);
    }
    return up;
}

uint32_t Conv::sample_repeat(uint32_t p, int32_t offset, uint32_t size)
{
    int32_t ip = static_cast<int32_t>(p) + offset;
    if(ip < 0) {
        uint32_t t = static_cast<uint32_t>(-ip);
        assert(t < size);
        return size - t;
    }
    uint32_t up = static_cast<uint32_t>(ip);
    if(size <= up) {
        uint32_t dp = up - size + 1;
        assert(dp < size);
        return dp;
    }
    return up;
}

//--- Linear
//-------------------------------------------------------------------
Linear::Linear()
    : input_features_(0)
    , output_features_(0)
    , weights_(nullptr)
    , bias_(nullptr)
{
}

Linear::Linear(Linear&& other) noexcept
    : input_features_(other.input_features_)
    , output_features_(other.output_features_)
    , weights_(other.weights_)
    , bias_(other.bias_)
{
    other.input_features_ = 0;
    other.output_features_ = 0;
    other.weights_ = nullptr;
    other.bias_ = nullptr;
}

Linear::Linear(uint32_t input_features, uint32_t output_features)
    : input_features_(input_features)
    , output_features_(output_features)
    , weights_(nullptr)
    , bias_(nullptr)
{
    uint32_t weights_size = input_features * output_features;
    uint32_t bias_size = output_features;
    weights_ = static_cast<float*>(d2p2_malloc(sizeof(float) * (weights_size + bias_size)));
    bias_ = weights_ + weights_size;
}

Linear::~Linear() noexcept
{
    d2p2_free(weights_);
    input_features_ = 0;
    output_features_ = 0;
    weights_ = nullptr;
    bias_ = nullptr;
}

Tensor Linear::operator()(const Tensor& tensor) const
{
    // weights: out_channels x in_channels x kernel_size
    Tensor result(tensor.rows(), output_features_, tensor.channels());
    result.setZeros();

    for(uint32_t c = 0; c < tensor.channels(); ++c) {
        for(uint32_t sr = 0; sr < tensor.rows(); ++sr) {
            for(uint32_t dc = 0; dc < output_features_; ++dc) {
                float t=0.0f;
                for(uint32_t sc = 0; sc < tensor.cols(); ++sc) {
                    float v = tensor(sr, sc, c);
                    float w = weights_[dc * input_features_ + sc];
                    t += v*w;
                }
                result(sr, dc, c) = t;
            }
        }
    }
    return result;
}

void Linear::weights(std::initializer_list<float> args)
{
    assert(args.size() == (input_features_ * output_features_));
    uint32_t i=0;
    for(auto&& itr = args.begin(); itr!= args.end(); ++itr,++i){
        weights_[i] = *itr;
    }
}

void Linear::bias(std::initializer_list<float> args)
{
    assert(args.size() == output_features_);
    uint32_t i=0;
    for(auto&& itr = args.begin(); itr!= args.end(); ++itr,++i){
        bias_[i] = *itr;
    }
}

//--- Conv1d
//-------------------------------------------------------------------
Conv1d::Conv1d()
    : input_channels_(0)
    , output_channels_(0)
    , kernel_size_(0)
    , stride_(1)
    , padding_(0)
    , padding_mode_(Conv::PaddingMode::Zeros)
    , weights_(nullptr)
    , bias_(nullptr)
{
}

Conv1d::Conv1d(Conv1d&& other) noexcept
    : input_channels_(other.input_channels_)
    , output_channels_(other.output_channels_)
    , kernel_size_(other.kernel_size_)
    , stride_(other.stride_)
    , padding_(other.padding_)
    , padding_mode_(other.padding_mode_)
    , weights_(other.weights_)
    , bias_(other.bias_)
{
    other.input_channels_ = 0;
    other.output_channels_ = 0;
    other.kernel_size_ = 0;
    other.stride_ = 0;
    other.padding_ = 0;
    other.padding_mode_ = Conv::PaddingMode::Zeros;
    other.weights_ = nullptr;
    other.bias_ = nullptr;
}

Conv1d::Conv1d(uint32_t input_channels, uint32_t output_channels, uint32_t kernel_size, uint32_t stride, uint32_t padding, Conv::PaddingMode padding_mode)
    : input_channels_(input_channels)
    , output_channels_(output_channels)
    , kernel_size_(kernel_size)
    , stride_(stride)
    , padding_(padding)
    , padding_mode_(padding_mode)
    , weights_(nullptr)
    , bias_(nullptr)
{
    uint32_t weights_size = input_channels_ * output_channels_ * kernel_size_;
    uint32_t bias_size = output_channels_;
    weights_ = static_cast<float*>(d2p2_malloc(sizeof(float) * (weights_size + bias_size)));
    bias_ = weights_ + weights_size;
}

Conv1d::~Conv1d() noexcept
{
    d2p2_free(weights_);
    input_channels_ = 0;
    output_channels_ = 0;
    kernel_size_ = 0;
    stride_ = 0;
    padding_mode_ = Conv::PaddingMode::Zeros;
    weights_ = nullptr;
    bias_ = nullptr;
}

Tensor Conv1d::operator()(const Tensor& tensor) const
{
    assert(input_channels_ == tensor.rows());
    assert(1 == tensor.cols());
    std::function<uint32_t(uint32_t, int32_t, uint32_t)> wrap_mode;
    switch(padding_mode_) {
    case Conv::PaddingMode::Reflect:
        wrap_mode = Conv::sample_reflect;
        break;
    case Conv::PaddingMode::Replicate:
        wrap_mode = Conv::sample_replicate;
        break;
    case Conv::PaddingMode::Repeat:
        wrap_mode = Conv::sample_repeat;
        break;
    default:
        wrap_mode = Conv::sample_zeros;
        break;
    }

    // weights: out_channels x in_channels x kernel_size
    Tensor result(src2dst_size(tensor.rows(), kernel_size_, stride_, padding_), 1, output_channels_);
    result.setZeros();
    auto [left, right] = Conv::kernel_range(kernel_size_, padding_, tensor.channels());
    auto [start, end] = Conv::conv_range(kernel_size_, padding_, tensor.channels());
    for(uint32_t oi = 0; oi < output_channels_; ++oi) {
        for(uint32_t ii = 0; ii < input_channels_; ++ii) {
            for(uint32_t c = start; c <= end; c += stride_) {
                float x = 0.0f;
                for(int32_t f = left; f <= right; ++f) {
                    uint32_t p = c + f;
                    float v = tensor.get2d(ii, 0, p);
                    float w = weight(oi, ii, static_cast<uint32_t>(f - left));
                    x += w * v;
                } // for(int32_t p
                uint32_t out_r = src2dst_index(c, kernel_size_, stride_, padding_);
                result(out_r, 0, oi) += x + bias_[oi];
            }     // for(uint32_t c
        }         // for(uint32_t ii
    } // for(uint32_t oi
    return result;
}

void Conv1d::weights(std::initializer_list<float> args)
{
    assert(args.size() == (input_channels_ * output_channels_ * kernel_size_));
    uint32_t i=0;
    for(auto&& itr = args.begin(); itr!= args.end(); ++itr,++i){
        weights_[i] = *itr;
    }
}

void Conv1d::bias(std::initializer_list<float> args)
{
    assert(args.size() == output_channels_);
    uint32_t i=0;
    for(auto&& itr = args.begin(); itr!= args.end(); ++itr,++i){
        bias_[i] = *itr;
    }
}

const float& Conv1d::weight(uint32_t out, uint32_t in, uint32_t kernel) const
{
    return weights_[(out*input_channels_ + in)*kernel_size_ + kernel];
}

//--- ConvTranspose1d
//-------------------------------------------------------------------
ConvTranspose1d::ConvTranspose1d()
    : input_channels_(0)
    , output_channels_(0)
    , kernel_size_(0)
    , stride_(1)
    , padding_(0)
    , padding_mode_(Conv::PaddingMode::Zeros)
    , weights_(nullptr)
    , bias_(nullptr)
{
}

ConvTranspose1d::ConvTranspose1d(ConvTranspose1d&& other) noexcept
    : input_channels_(other.input_channels_)
    , output_channels_(other.output_channels_)
    , kernel_size_(other.kernel_size_)
    , stride_(other.stride_)
    , padding_(other.padding_)
    , padding_mode_(other.padding_mode_)
    , weights_(other.weights_)
    , bias_(other.bias_)
{
    other.input_channels_ = 0;
    other.output_channels_ = 0;
    other.kernel_size_ = 0;
    other.stride_ = 0;
    other.padding_ = 0;
    other.padding_mode_ = Conv::PaddingMode::Zeros;
    other.weights_ = nullptr;
    other.bias_ = nullptr;
}

ConvTranspose1d::ConvTranspose1d(uint32_t input_channels, uint32_t output_channels, uint32_t kernel_size, uint32_t stride, uint32_t padding, Conv::PaddingMode padding_mode)
    : input_channels_(input_channels)
    , output_channels_(output_channels)
    , kernel_size_(kernel_size)
    , stride_(stride)
    , padding_(padding)
    , padding_mode_(padding_mode)
    , weights_(nullptr)
    , bias_(nullptr)
{
    uint32_t weights_size = input_channels_ * output_channels_ * kernel_size_;
    uint32_t bias_size = output_channels_;
    weights_ = static_cast<float*>(d2p2_malloc(sizeof(float) * (weights_size + bias_size)));
    bias_ = weights_ + weights_size;
}

ConvTranspose1d::~ConvTranspose1d() noexcept
{
    d2p2_free(weights_);
    input_channels_ = 0;
    output_channels_ = 0;
    kernel_size_ = 0;
    stride_ = 0;
    padding_mode_ = Conv::PaddingMode::Zeros;
    weights_ = nullptr;
    bias_ = nullptr;
}

Tensor ConvTranspose1d::operator()(const Tensor& tensor) const
{
    assert(input_channels_ == tensor.channels());
    assert(1 == tensor.cols());
    std::function<uint32_t(uint32_t, int32_t, uint32_t)> wrap_mode;
    switch(padding_mode_) {
    case Conv::PaddingMode::Reflect:
        wrap_mode = Conv::sample_reflect;
        break;
    case Conv::PaddingMode::Replicate:
        wrap_mode = Conv::sample_replicate;
        break;
    case Conv::PaddingMode::Repeat:
        wrap_mode = Conv::sample_repeat;
        break;
    default:
        wrap_mode = Conv::sample_zeros;
        break;
    }

    // weights: out_channels x in_channels x kernel_size
    Tensor result(transpose_src2dst_size(tensor.rows(), kernel_size_, stride_, padding_), 1, output_channels_);
    result.setZeros();
    auto [left, right] = Conv::kernel_range(kernel_size_, padding_, tensor.rows());
    auto [start, end] = Conv::conv_range(kernel_size_, padding_, tensor.rows());
    for(uint32_t oi = 0; oi < output_channels_; ++oi) {
        for(uint32_t ii = 0; ii < input_channels_; ++ii) {
            for(uint32_t c = start; c <= end; c += stride_) {
                float x = 0.0f;
                for(int32_t f = left; f <= right; ++f) {
                    uint32_t p = c + f;
                    float v = tensor.get2d(ii, 0, p);
                    float w = weight(oi, ii, static_cast<uint32_t>(f - left));
                    x += w * v;
                } // for(int32_t p
                uint32_t out_r = src2dst_index(c, kernel_size_, stride_, padding_);
                result(out_r, 0, oi) += x + bias_[oi];
            }     // for(uint32_t c
        }         // for(uint32_t ii
    } // for(uint32_t oi
    return result;
}

void ConvTranspose1d::weights(std::initializer_list<float> args)
{
    assert(args.size() == (input_channels_ * output_channels_ * kernel_size_));
    uint32_t i=0;
    for(auto&& itr = args.begin(); itr!= args.end(); ++itr,++i){
        weights_[i] = *itr;
    }
}

void ConvTranspose1d::bias(std::initializer_list<float> args)
{
    assert(args.size() == output_channels_);
    uint32_t i=0;
    for(auto&& itr = args.begin(); itr!= args.end(); ++itr,++i){
        bias_[i] = *itr;
    }
}

const float& ConvTranspose1d::weight(uint32_t out, uint32_t in, uint32_t kernel) const
{
    return weights_[(out*input_channels_ + in)*kernel_size_ + kernel];
}

//--- Conv2d
//-------------------------------------------------------------------
Conv2d::Conv2d()
    : input_channels_(0)
    , output_channels_(0)
    , kernel_size_(0)
    , stride_(1)
    , padding_(0)
    , padding_mode_(Conv::PaddingMode::Zeros)
    , weights_(nullptr)
    , bias_(nullptr)
{
}

Conv2d::Conv2d(Conv2d&& other) noexcept
    : input_channels_(other.input_channels_)
    , output_channels_(other.output_channels_)
    , kernel_size_(other.kernel_size_)
    , stride_(other.stride_)
    , padding_(other.padding_)
    , padding_mode_(other.padding_mode_)
    , weights_(other.weights_)
    , bias_(other.bias_)
{
    other.input_channels_ = 0;
    other.output_channels_ = 0;
    other.kernel_size_ = 0;
    other.stride_ = 0;
    other.padding_ = 0;
    other.padding_mode_ = Conv::PaddingMode::Zeros;
    other.weights_ = nullptr;
    other.bias_ = nullptr;
}

Conv2d::Conv2d(uint32_t input_channels, uint32_t output_channels, uint32_t kernel_size, uint32_t stride, uint32_t padding, Conv::PaddingMode padding_mode, bool enable_bias)
    : input_channels_(input_channels)
    , output_channels_(output_channels)
    , kernel_size_(kernel_size)
    , stride_(stride)
    , padding_(padding)
    , padding_mode_(padding_mode)
    , weights_(nullptr)
    , bias_(nullptr)
{
    uint32_t weights_size = input_channels_ * output_channels_ * kernel_size_ * kernel_size_;
    uint32_t bias_size = output_channels_;
    weights_ = static_cast<float*>(d2p2_malloc(sizeof(float) * (weights_size + bias_size)));
    bias_ = weights_ + weights_size;
}

Conv2d::~Conv2d() noexcept
{
    d2p2_free(weights_);
    input_channels_ = 0;
    output_channels_ = 0;
    kernel_size_ = 0;
    stride_ = 0;
    padding_ = 0;
    padding_mode_ = Conv::PaddingMode::Zeros;
    weights_ = nullptr;
    bias_ = nullptr;
}

Tensor Conv2d::operator()(const Tensor& tensor) const
{
    std::function<uint32_t(uint32_t, int32_t, uint32_t)> wrap_mode;
    switch(padding_mode_) {
    case Conv::PaddingMode::Reflect:
        wrap_mode = Conv::sample_reflect;
        break;
    case Conv::PaddingMode::Replicate:
        wrap_mode = Conv::sample_replicate;
        break;
    case Conv::PaddingMode::Repeat:
        wrap_mode = Conv::sample_repeat;
        break;
    default:
        wrap_mode = Conv::sample_zeros;
        break;
    }

    //weight = out_channels x input_channels x kernel_size x kernel_size
    uint32_t orows = src2dst_size(tensor.rows(), kernel_size_, stride_, padding_);
    uint32_t ocols = src2dst_size(tensor.cols(), kernel_size_, stride_, padding_);
    Tensor result(orows, ocols, output_channels_);
    result.setZeros();
    auto [rleft, rright] = Conv::kernel_range(kernel_size_, padding_, tensor.rows());
    auto [rstart, rend] = Conv::conv_range(kernel_size_, padding_, tensor.rows());
    auto [cleft, cright] = Conv::kernel_range(kernel_size_, padding_, tensor.cols());
    auto [cstart, cend] = Conv::conv_range(kernel_size_, padding_, tensor.cols());

    for(uint32_t oi = 0; oi < output_channels_; ++oi) {
        for(uint32_t ii = 0; ii < input_channels_; ++ii) {
            for(uint32_t r = rstart; r <= rend; r += stride_) {
                uint32_t out_r = src2dst_index(r, kernel_size_, stride_, padding_);
                for(uint32_t c = cstart; c <= cend; c += stride_) {
                    uint32_t out_c = src2dst_index(c, kernel_size_, stride_, padding_);
                    float x = 0.0f;
                    for(int32_t rf = rleft; rf <= rright; ++rf) {
                        uint32_t rp = r + rf;
                        for(int32_t cf = cleft; cf <= cright; ++cf) {
                            uint32_t cp = c + cf;
                            float v = tensor.get2d(rp, cp, ii);
                            float w = weight(oi, ii, static_cast<uint32_t>(rf - rleft), static_cast<uint32_t>(cf - cleft));
                            x += w * v;
                        } // for(int32_t cf
                    }     // for(int32_t rf
                    result(out_r, out_c, oi) += x + bias_[oi];
                }         // for(uint32_t c
            } // for(uint32_t r
        }     // for(uint32_t ii
    }         // for(uint32_t oi
    return result;
}

void Conv2d::weights(std::initializer_list<float> args)
{
    assert(args.size() == (input_channels_ * output_channels_ * kernel_size_ * kernel_size_));
    uint32_t i=0;
    for(auto&& itr = args.begin(); itr!= args.end(); ++itr,++i){
        weights_[i] = *itr;
    }
}

void Conv2d::bias(std::initializer_list<float> args)
{
    assert(args.size() == output_channels_);
    uint32_t i=0;
    for(auto&& itr = args.begin(); itr!= args.end(); ++itr,++i){
        bias_[i] = *itr;
    }
}

const float& Conv2d::weight(uint32_t out, uint32_t in, uint32_t kr, uint32_t kc) const
{
    return weights_[((out*input_channels_ + in)*kernel_size_ + kr)*kernel_size_+kc];
}

} // namespace d2p2
