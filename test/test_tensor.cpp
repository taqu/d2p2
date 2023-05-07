#include "catch_amalgamated.hpp"
#include "catch_wrap.hpp"
#include "d2p2/tensor.h"
#include "src/ispc/matrix.ispc.h"
#include <iostream>

TEST_CASE("Mul")
{
    using namespace d2p2;
    {
        float A[] = {
            1, 2,
            3, 4,
            5, 6};
        float B[] = {
            7,
            8};
        float R[3];
        ispc::matrix_mul(R, 3, 2, 1, A, B);
        EXPECT_FLOAT_EQ(23.0f, R[0]);
        EXPECT_FLOAT_EQ(53.0f, R[1]);
        EXPECT_FLOAT_EQ(83.0f, R[2]);
    }
    {
        Tensor W1(2, 3, 1, std::initializer_list<float>{1, 2, 3, 4, 5, 6});
        Tensor x(1, 2, 1, {7, 8});
        Tensor a = x * W1;
        EXPECT_FLOAT_EQ(39.0f, a(0, 0, 0));
        EXPECT_FLOAT_EQ(54.0f, a(0, 1, 0));
        EXPECT_FLOAT_EQ(69.0f, a(0, 2, 0));

        std::cout << a << std::endl;
    }
}

TEST_CASE("Add")
{
    float A[] = {1, 2, 3, 4};
    float B[] = {5, 6, 7, 8};
    float R[4];
    ispc::matrix_add(R, 4, A, B);
    EXPECT_FLOAT_EQ(6.0f, R[0]);
    EXPECT_FLOAT_EQ(8.0f, R[1]);
    EXPECT_FLOAT_EQ(10.0f, R[2]);
    EXPECT_FLOAT_EQ(12.0f, R[3]);
}

TEST_CASE("Muladd")
{
    float A[] = {
        1, 2,
        3, 4,
        5, 6
    };
    float B[] = {7, 8};
    float C[] = {1, 2, 3};
    float R[3];
    ispc::matrix_muladd(R, 3, 2, 1, A, B, C);
    EXPECT_FLOAT_EQ(24.0f, R[0]);
    EXPECT_FLOAT_EQ(55.0f, R[1]);
    EXPECT_FLOAT_EQ(86.0f, R[2]);
}

TEST_CASE("Softmax")
{
    using namespace d2p2;
    Tensor a(1, 3, 1, std::initializer_list<float>{0.3f, 2.9f, 4.0f});
    Tensor r = softmax(a);
    std::cout << r << std::endl;
}

