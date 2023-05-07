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
#include "d2p2/d2p2.h"
#include <cstdlib>
#include <random>
#include "d2p2/img/cppimg.h"

namespace d2p2
{
namespace
{
    /**
     * @brief 32 bit right rotation
     * @param [in] x ... input
     * @param [in] r ... count of rotation
     * @return rotated
     */
    inline uint32_t rotr32(uint32_t x, uint32_t r)
    {
        return (x >> r) | (x << ((~r + 1) & 31U));
    }

    /**
     * @brief Convert to a [0 1) real number
     * @param [in] x
     * @return a [0 1) real number
     */
    inline float to_real32(uint32_t x)
    {
        return static_cast<float>((x >> 8) * (1.0 / 16777216.0));
    }

    //--- SplitMix
    //---------------------------------------------------------
    /**
     * @brief A fast 64 bit PRNG
     *
     * | Feature |      |
     * | :------ | :--- |
     * | Bits    | 64   |
     * | Period  | 2^64 |
     * | Streams | 1    |
     */
    class SplitMix
    {
    public:
        static uint64_t next(uint64_t& state);
    };

    uint64_t SplitMix::next(uint64_t& state)
    {
        state += 0x9E3779B97f4A7C15ULL;
        uint64_t t = state;
        t = (t ^ (t >> 30)) * 0xBF58476D1CE4E5B9ULL;
        t = (t ^ (t >> 27)) * 0x94D049BB133111EBULL;
        return t ^ (t >> 31);
    }

} // namespace

void* d2p2_malloc(size_t size)
{
    return ::malloc(size);
}

void d2p2_free(void* ptr)
{
    return ::free(ptr);
}

PCGS32::PCGS32()
    : state_(12345ULL)
    , increment_(DefaultStream)
{
}

PCGS32::PCGS32(uint64_t seed, uint64_t stream)
{
    srand(seed);
    increment_ = stream | 1ULL;
}

PCGS32::~PCGS32()
{
}

void PCGS32::srand(uint64_t seed, uint64_t stream)
{
    state_ = SplitMix::next(seed);
    while(0 == state_) {
        state_ = SplitMix::next(state_);
    }
    increment_ = stream | 1ULL;
}

uint32_t PCGS32::rand()
{
    uint64_t x = state_;
    uint32_t c = static_cast<uint32_t>(x >> 59);
    state_ = x * Multiplier + increment_;
    x ^= x >> 18;
    return rotr32(static_cast<uint32_t>(x >> 27), c);
}

float PCGS32::frand()
{
    return to_real32(rand());
}

//--- Context
//----------------------------------------------
Context Context::context_;

Context::Context()
{
}

Context::~Context()
{
}

Context& Context::get()
{
    return context_;
}

PCGS32& Context::random()
{
    return random_;
}

void Context::resetRandom(bool resetStream)
{
    std::random_device device;
    if(resetStream){
        random_.srand(device(),device());
    }else{
        random_.srand(device());
    }
}
} // namespace d2p2