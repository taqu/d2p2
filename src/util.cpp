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
#include "d2p2/util.h"
#include <cstdio>
#include <random>
#include "d2p2/tensor.h"
#include "ispc/matrix.ispc.h"

#ifdef _WIN32
#    include <Windows.h>
#endif

#define CPPIMG_IMPLEMENTATION
#include "d2p2/img/cppimg.h"
#define SZLIB_IMPLEMENTATION
#include "d2p2/img/szlib.h"

namespace d2p2
{
namespace
{
    class IOStream: public cppimg::Stream
    {
    public:
        IOStream();
        ~IOStream();
        bool open(const char8_t* path);
        void close();
        virtual bool valid() const;
        virtual bool seek(cppimg::off_t pos, cppimg::s32 whence);
        virtual cppimg::off_t tell();
        virtual cppimg::s64 size();

        virtual cppimg::s32 read(size_t size, void* dst);
        virtual cppimg::s32 write(size_t size, const void* dst);
    private:
        FILE* file_;
    };

    IOStream::IOStream()
        :file_(nullptr)
    {
    }

        IOStream::~IOStream()
    {
            close();
    }

        bool IOStream::open(const char8_t* path)
    {
            close();
            #ifdef _WIN32
            if(0 != fopen_s(&file_, reinterpret_cast<const char*>(path), "rb")){
                return false;
            }
            #else
            file_ = fopen(reinterpret_cast<const char*>(path), "rb");
            if(nullptr == file_){
                return false;
        }
            #endif
            return true;
    }

        void IOStream::close()
    {
            if(nullptr == file_){
                return;
            }
            fclose(file_);
            file_ = nullptr;
    }


        bool IOStream::valid() const
    {
            return nullptr != file_;
    }

        bool IOStream::seek(cppimg::off_t pos, cppimg::s32 whence)
    {
            return 0 == fseek(file_, pos, whence);
    }

        cppimg::off_t IOStream::tell()
    {
            return ftell(file_);
    }

        cppimg::s64 IOStream::size()
    {
            struct _stat64 s;
            if(0 != _fstat64(fileno(file_), &s)){
                return 0;
            }
            return s.st_size;
    }

        cppimg::s32 IOStream::read(size_t size, void* dst)
    {
            return fread(dst, size, 1, file_);
    }

        cppimg::s32 IOStream::write(size_t size, const void* dst)
    {
            return fwrite(dst, size, 1, file_);
    }

} // namespace

//--- Image
Image::Image()
    : width_(0)
    , height_(0)
    , channels_(0)
    , data_(nullptr)
{
}

Image::Image(Image&& other) noexcept
    : width_(other.width_)
    , height_(other.height_)
    , channels_(other.channels_)
    , data_(other.data_)
{
    other.width_ = 0;
    other.height_ = 0;
    other.channels_ = 0;
    other.data_ = nullptr;
}

Image::Image(uint32_t width, uint32_t height, uint32_t channels, uint8_t* data)
    : width_(width)
    , height_(height)
    , channels_(channels)
    , data_(data)
{
}

Image::~Image()
{
    width_ = 0;
    height_ = 0;
    channels_ = 0;
    d2p2::d2p2_free(data_);
    data_ = nullptr;
}

uint32_t Image::width() const
{
    return width_;
}

uint32_t Image::height() const
{
    return height_;
}

uint32_t Image::channels() const
{
    return channels_;
}

const uint8_t& Image::operator()(uint32_t r, uint32_t c) const
{
    return data_[(r * width_ + c) * channels_];
}

uint8_t& Image::operator()(uint32_t r, uint32_t c)
{
    return data_[(r * width_ + c) * channels_];
}

Image::operator const uint8_t*() const
{
    return data_;
}

Image::operator uint8_t*()
{
    return data_;
}

Image& Image::operator=(Image&& other) noexcept
{
    if(this == &other) {
        return *this;
    }
    d2p2::d2p2_free(data_);
    width_ = other.width_;
    height_ = other.height_;
    channels_ = other.channels_;
    data_ = other.data_;

    other.width_ = 0;
    other.height_ = 0;
    other.channels_ = 0;
    other.data_ = nullptr;
    return *this;
}

std::vector<std::filesystem::path> parse_directory(const char* path, std::function<bool(const std::filesystem::directory_entry&)> predicate, bool shuffle)
{
    assert(nullptr != path);
    std::vector<std::filesystem::path> files;
    for(const std::filesystem::directory_entry& entry: std::filesystem::recursive_directory_iterator(path)) {
        if(!entry.is_regular_file()) {
            continue;
        }
        if(predicate(entry)) {
            if(files.capacity() <= files.size()) {
                files.reserve(files.size() + 2048);
            }
            files.push_back(std::filesystem::canonical(entry.path()));
        }
    }
    if(shuffle) {
        std::random_device seed_gen;
        std::mt19937 engine(seed_gen());
        std::shuffle(files.begin(), files.end(), engine);
    }
    return files;
}

namespace
{
    void to_matrix(Tensor& matrix, uint32_t width, uint32_t height, const uint8_t* data)
    {
        matrix = std::move(Tensor(static_cast<int32_t>(height), static_cast<int32_t>(width), 1));
        ispc::array_to_float(width*height, (float*)matrix, data, 1.0f/255.0f);
    }
}

bool load_image(Tensor& image, const std::filesystem::path& path)
{
    IOStream file;
    {
        const char8_t* u8path = path.u8string().c_str();
        if(!file.open(u8path)){
            return false;
        }
    }

    std::string ext(reinterpret_cast<const char*>(path.extension().c_str()));
    cppimg::s32 width = 0, height = 0;
    cppimg::ColorType colorType = cppimg::ColorType::RGB;
    uint8_t* bytes = nullptr;
    if(".png" == ext) {
        if(!cppimg::PNG::read(width, height, colorType, nullptr, file)){
            return false;
        }
        bytes = static_cast<uint8_t*>(d2p2_malloc(sizeof(uint8_t)*width*height*cppimg::getBytesPerPixel(colorType)));
        if(!cppimg::PNG::read(width, height, colorType, bytes, file)){
            d2p2_free(bytes);
            return false;
        }
    }else if(".jpg" == ext){
        if(!cppimg::JPEG::read(width, height, colorType, nullptr, file)){
            return false;
        }
        bytes = static_cast<uint8_t*>(d2p2_malloc(sizeof(uint8_t)*width*height*cppimg::getBytesPerPixel(colorType)));
        if(!cppimg::JPEG::read(width, height, colorType, bytes, file)){
            d2p2_free(bytes);
            return false;
        }
    }else{
        return false;
    }

    switch(colorType){
    case cppimg::ColorType::GRAY:{
        uint8_t* dst = static_cast<uint8_t*>(d2p2::d2p2_malloc(sizeof(uint8_t) * width * height * 3));
        cppimg::convertGrayToRGB(width, height, dst, bytes);
        d2p2_free(bytes);
        bytes = dst;
    }
        break;
    case cppimg::ColorType::RGBA:{
        uint8_t* dst = static_cast<uint8_t*>(d2p2::d2p2_malloc(sizeof(uint8_t) * width * height * 3));
        cppimg::convertRGBAToRGB(width, height, dst, bytes);
        d2p2_free(bytes);
        bytes = dst;
    }
        break;
    default:
        break;
    }
    float* m = static_cast<float*>(d2p2_malloc(sizeof(float)*width*height*3));
    for(int32_t i=0; i<(width*height*3); ++i){
        m[i] = cppimg::Color::toFloat(bytes[i]);
    }
    d2p2_free(bytes);
    image = std::move(Tensor(width, height, 3, m));
    return true;
}

bool load_image_gray(Tensor& image, const std::filesystem::path& path)
{
    IOStream file;
    {
        const char8_t* u8path = path.u8string().c_str();
        if(!file.open(u8path)){
            return false;
        }
    }

    std::string ext(reinterpret_cast<const char*>(path.extension().c_str()));
    cppimg::s32 width = 0, height = 0;
    cppimg::ColorType colorType = cppimg::ColorType::RGB;
    uint8_t* bytes = nullptr;
    if(".png" == ext) {
        if(!cppimg::PNG::read(width, height, colorType, nullptr, file)){
            return false;
        }
        bytes = static_cast<uint8_t*>(d2p2_malloc(sizeof(uint8_t)*width*height*cppimg::getBytesPerPixel(colorType)));
        if(!cppimg::PNG::read(width, height, colorType, bytes, file)){
            d2p2_free(bytes);
            return false;
        }
    }else if(".jpg" == ext){
        if(!cppimg::JPEG::read(width, height, colorType, nullptr, file)){
            return false;
        }
        bytes = static_cast<uint8_t*>(d2p2_malloc(sizeof(uint8_t)*width*height*cppimg::getBytesPerPixel(colorType)));
        if(!cppimg::JPEG::read(width, height, colorType, bytes, file)){
            d2p2_free(bytes);
            return false;
        }
    }else{
        return false;
    }

    switch(colorType){
    case cppimg::ColorType::RGB: {
        uint8_t* dst = static_cast<uint8_t*>(d2p2::d2p2_malloc(sizeof(uint8_t) * width * height));
        cppimg::convertRGBToGray(width, height, dst, bytes);
        d2p2_free(bytes);
        bytes = dst;
    }
        break;
    case cppimg::ColorType::RGBA:{
        uint8_t* dst = static_cast<uint8_t*>(d2p2::d2p2_malloc(sizeof(uint8_t) * width * height));
        cppimg::convertRGBAToGray(width, height, dst, bytes);
        d2p2_free(bytes);
        bytes = dst;
    }
        break;
    default:
        break;
    }
    float* m = static_cast<float*>(d2p2_malloc(sizeof(float)*width*height*3));
    for(int32_t i=0; i<(width*height*3); ++i){
        m[i] = cppimg::Color::toFloat(bytes[i]);
    }
    d2p2_free(bytes);
    image = std::move(Tensor(width, height, 1, m));
    return true;
}
} // namespace d2p2
