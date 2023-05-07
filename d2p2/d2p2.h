#ifndef INC_D2P2_H_
#define INC_D2P2_H_
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
#include <cassert>
#include <cstdint>

namespace d2p2
{
void* d2p2_malloc(size_t size);
void d2p2_free(void* ptr);
} // namespace d2p2

#define SZ_MALLOC(size) d2p2::d2p2_malloc((size))
#define SZ_FREE(ptr) d2p2::d2p2_free((ptr))
#define CPPIMG_MALLOC(size) d2p2::d2p2_malloc((size))
#define CPPIMG_FREE(ptr) d2p2::d2p2_free((ptr))
#include "img/cppimg.h"

namespace d2p2
{
class PCGS32
{
public:
    inline static constexpr uint64_t DefaultStream = 1442695040888963407ULL;
    /**
     * @brief Initialize with CPPRNG_DEFAULT_SEED64 and DefaultStream
     */
    PCGS32();

    /**
     * @brief Initialize with a seed and a stream
     * @param [in] seed ... initialize states with
     * @param [in] stream
     */
    explicit PCGS32(uint64_t seed, uint64_t stream = DefaultStream);
    ~PCGS32();

    /**
     * @brief Initialize with a seed and a stream
     * @param [in] seed ... initialize states with
     * @param [in] stream
     */
    void srand(uint64_t seed, uint64_t stream = DefaultStream);

    /**
     * @brief Generate a 32bit unsigned value
     * @return
     */
    uint32_t rand();

    /**
     * @brief Generate a 32bit real number
     * @return [0 1)
     */
    float frand();

private:
    inline static constexpr uint64_t Multiplier = 6364136223846793005ULL;
    uint64_t increment_;
    uint64_t state_;
};

class Context
{
public:
    static Context& get();

    PCGS32& random();
    void resetRandom(bool resetStream=true);
private:
    static Context context_;
    Context();
    ~Context();

    PCGS32 random_;
};
} // namespace d2p2
#endif // INC_D2P2_H_
