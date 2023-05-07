#include "catch_wrap.hpp"
#include "d2p2/tensor.h"
#include "d2p2/function.h"
#include "src/ispc/matrix.ispc.h"
#include <iostream>

TEST_CASE("Conv1d")
{
    using namespace d2p2;
    #if 1
    {
        Tensor input(2, 1, 3, {
            1.0f, 2.0f, 3.0f,
            4.0f, 5.0f, 6.0f,
        });

        Conv1d conv1d(2, 3, 3);
        conv1d.weights({ //3 2 3
            1.0f, 1.0f, 1.0f,
            2.0f, 2.0f, 2.0f,
            3.0f, 3.0f, 3.0f,
            4.0f, 4.0f, 4.0f,
            5.0f, 5.0f, 5.0f,
            6.0f, 6.0f, 6.0f,
        });
        conv1d.bias({0.0f, 0.0f, 0.0f});
        Tensor result = conv1d(input);
        std::cout << result << std::endl;
    }
    #endif

    #if 1
    {
        Tensor input(4, 1, 6, {
            1.f,  2.f,  3.f,  4.f,  5.f,  6.f,
            7.f,  8.f,  9.f, 10.f, 11.f, 12.f,
            13.f, 14.f, 15.f, 16.f, 17.f, 18.f,
            19.f, 20.f, 21.f, 22.f, 23.f, 24.f});

        Conv1d conv1d(4, 5, 6);
        conv1d.weights({ //5 4 6
            1.f,  1.f,  1.f,  1.f,  1.f,  1.f,
            2.f,  2.f,  2.f,  2.f,  2.f,  2.f,
            3.f,  3.f,  3.f,  3.f,  3.f,  3.f,
            4.f,  4.f,  4.f,  4.f,  4.f,  4.f,

            5.f,  5.f,  5.f,  5.f,  5.f,  5.f,
            6.f,  6.f,  6.f,  6.f,  6.f,  6.f,
            7.f,  7.f,  7.f,  7.f,  7.f,  7.f,
            8.f,  8.f,  8.f,  8.f,  8.f,  8.f,

            9.f,  9.f,  9.f,  9.f,  9.f,  9.f,
            10.f, 10.f, 10.f, 10.f, 10.f, 10.f,
            11.f, 11.f, 11.f, 11.f, 11.f, 11.f,
            12.f, 12.f, 12.f, 12.f, 12.f, 12.f,

            13.f, 13.f, 13.f, 13.f, 13.f, 13.f,
            14.f, 14.f, 14.f, 14.f, 14.f, 14.f,
            15.f, 15.f, 15.f, 15.f, 15.f, 15.f,
            16.f, 16.f, 16.f, 16.f, 16.f, 16.f,

            17.f, 17.f, 17.f, 17.f, 17.f, 17.f,
            18.f, 18.f, 18.f, 18.f, 18.f, 18.f,
            19.f, 19.f, 19.f, 19.f, 19.f, 19.f,
            20.f, 20.f, 20.f, 20.f, 20.f, 20.f});

        conv1d.bias({0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
        Tensor result = conv1d(input);
        std::cout << result << std::endl;
    }
    #endif
}

TEST_CASE("ConvTranspose1d")
{
    using namespace d2p2;
    #if 1
    {
        Tensor input(3, 1, 2, {
            1.0f, 2.0f, 3.0f,
            4.0f, 5.0f, 6.0f,
        });

        ConvTranspose1d conv1d(2, 4, 3);
        conv1d.weights({ //2 4 3
            1.0f, 1.0f, 1.0f,
            2.0f, 2.0f, 2.0f,
            3.0f, 3.0f, 3.0f,
            4.0f, 4.0f, 4.0f,
            5.0f, 5.0f, 5.0f,
            6.0f, 6.0f, 6.0f,
            7.0f, 7.0f, 7.0f,
            8.0f, 8.0f, 8.0f,
        });
        conv1d.bias({0.0f, 0.0f, 0.0f, 0.0f});
        Tensor result = conv1d(input);
        std::cout << result << std::endl;
    }
    #endif
}

TEST_CASE("Conv2d")
{
    using namespace d2p2;
    #if 1
    {
        Tensor input(3, 4, 3, {
            1.f, 2.f, 3.f, 4.f,
            5.f, 6.f, 7.f, 8.f,
            9.f, 10.f, 11.f, 12.f,

            13.f, 14.f, 15.f, 16.f,
            17.f, 18.f, 19.f, 20.f,
            21.f, 22.f, 23.f, 24.f,

            25.f, 26.f, 27.f, 28.f,
            29.f, 30.f, 31.f, 32.f,
            33.f, 34.f, 35.f, 36.f,
        });
        //print_numpy(std::cout, input) << std::endl;

        Conv2d conv2d(3, 4, 2);
        conv2d.weights({ //4 3 2 2
            #if 0
            1.f, 3.f, 2.f, 4.f,
            5.f, 7.f, 6.f, 8.f,
            9.f, 11.f, 10.f, 12.f,

            13.f, 15.f, 14.f, 16.f,
            17.f, 19.f, 18.f, 20.f,
            21.f, 23.f, 22.f, 24.f,

            25.f, 27.f, 26.f, 28.f,
            29.f, 31.f, 30.f, 32.f,
            33.f, 35.f, 34.f, 36.f,

            37.f, 39.f, 38.f, 40.f,
            41.f, 43.f, 42.f, 44.f,
            45.f, 47.f, 46.f, 48.f,
            #else
            1.f, 2.f, 3.f, 4.f,
            5.f, 6.f, 7.f, 8.f,
            9.f, 10.f, 11.f, 12.f,

            13.f, 14.f, 15.f, 16.f,
            17.f, 18.f, 19.f, 20.f,
            21.f, 22.f, 23.f, 24.f,

            25.f, 26.f, 27.f, 28.f,
            29.f, 30.f, 31.f, 32.f,
            33.f, 34.f, 35.f, 36.f,

            37.f, 38.f, 39.f, 40.f,
            41.f, 42.f, 43.f, 44.f,
            45.f, 46.f, 47.f, 48.f,
            #endif
        });
        conv2d.bias({0.0f, 0.0f, 0.0f, 0.0f});
        Tensor result = conv2d(input);
        print_numpy(std::cout, result) << std::endl;
    }
    #endif
}

TEST_CASE("Linear")
{
    using namespace d2p2;
    #if 1
    {
        Tensor input(2, 3, 4, {
            1.f, 2.f, 3.f,
            4.f, 5.f, 6.f,
            7.f, 8.f, 9.f,
            10.f, 11.f, 12.f,
            13.f, 14.f, 15.f,
            16.f, 17.f, 18.f,
            19.f, 20.f, 21.f,
            22.f, 23.f, 24.f,
        });
        std::cout << input << std::endl;

        Linear linear(3,3);
        linear.weights({ //3 3
            1.f, 2.f, 3.f,
            4.f, 5.f, 6.f,
            7.f, 8.f, 9.f,
        });
        linear.bias({0.0f, 0.0f, 0.0f});
        Tensor result = linear(input);
        print_numpy(std::cout, result) << std::endl;
    }
    #endif
}

