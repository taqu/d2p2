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
        std::cout << "---------"<< std::endl;
        Tensor input(1, 2, 3, {
            1.0f, 2.0f, 3.0f,
            4.0f, 5.0f, 6.0f,
        });
        std::cout << "input:" << input << std::endl;

        Conv1d conv1d(2, 3, 3);
        conv1d.weights({ //3 2 3
            1.0f, 2.0f, 3.0f,
            4.0f, 5.0f, 6.0f,
            1.0f, 2.0f, 3.0f,
            4.0f, 5.0f, 7.0f,
            1.0f, 2.0f, 3.0f,
            4.0f, 5.0f, 8.0f,
        });
        std::cout << "weights:"<< conv1d.weights() << std::endl;
        conv1d.bias({0.0f, 0.0f, 0.0f});
        Tensor result = conv1d(input);
        std::cout << "result:"<< result << std::endl;
    }
    #endif

    #if 1
    {
        std::cout << "---------"<< std::endl;
        Tensor input(4, 2, 6, {
            1,2,3,4,5,6,1,2,3,4,5,6,7,8,9,10,11,12,1,2,3,4,5,6,13,14,15,16,17,18,1,2,3,4,5,6,19,20,21,22,23,24,1,2,3,4,5,6,});
        std::cout << "input:" << input << std::endl;

        Conv1d conv1d(2, 5, 6);
        conv1d.weights({ //5 2 6
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

            //11.f, 11.f, 11.f, 11.f, 11.f, 11.f,
            //12.f, 12.f, 12.f, 12.f, 12.f, 12.f,
            //13.f, 13.f, 13.f, 13.f, 13.f, 13.f,
            //14.f, 14.f, 14.f, 14.f, 14.f, 14.f,
            //15.f, 15.f, 15.f, 15.f, 15.f, 15.f,
            //16.f, 16.f, 16.f, 16.f, 16.f, 16.f,

            //17.f, 17.f, 17.f, 17.f, 17.f, 17.f,
            //18.f, 18.f, 18.f, 18.f, 18.f, 18.f,
            //19.f, 19.f, 19.f, 19.f, 19.f, 19.f,
            //20.f, 20.f, 20.f, 20.f, 20.f, 20.f,
            });
        std::cout << "weights:"<< conv1d.weights() << std::endl;
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
        std::cout << "---------"<< std::endl;
        Tensor input(1, 2, 3, {
            1.0f, 2.0f, 3.0f,
            4.0f, 5.0f, 6.0f,
        });
        std::cout << "input:" << input << std::endl;

        ConvTranspose1d conv1d(2, 3, 3);
        conv1d.weights({ //2 3 3
            1.0f, 2.0f, 3.0f,
            4.0f, 5.0f, 6.0f,
            7.0f, 8.0f, 9.0f,
            1.0f, 2.0f, 3.0f,
            4.0f, 5.0f, 6.0f,
            7.0f, 8.0f, 10.0f,
        });
        std::cout << "weights:"<< conv1d.weights() << std::endl;
        conv1d.bias({0.0f, 0.0f, 0.0f});
        Tensor result = conv1d(input);
        std::cout << "result:"<< result << std::endl;
    }
    #endif

    #if 1
    {
        std::cout << "---------"<< std::endl;
        Tensor input(1, 2, 3, {
            1.0f, 2.0f, 3.0f,
            4.0f, 5.0f, 6.0f,
        });
        std::cout << "input:" << input << std::endl;

        ConvTranspose1d conv1d(2, 3, 2);
        conv1d.weights({ //2 3 2
            1.0f, 2.0f, 3.0f,
            4.0f, 5.0f, 6.0f,
            1.0f, 2.0f, 3.0f,
            4.0f, 5.0f, 7.0f,
        });
        std::cout << "weights:"<< conv1d.weights() << std::endl;
        conv1d.bias({0.0f, 0.0f, 0.0f});
        Tensor result = conv1d(input);
        std::cout << "result:"<< result << std::endl;
    }
    #endif
}

TEST_CASE("Conv2d")
{
    using namespace d2p2;
    #if 1
    {
        Tensor input(1, 1, 2, 2, {
            1.f, 2.f, 3.f, 4.f,
        });
        std::cout << "input:" << input << std::endl;

        Conv2d conv2d(1, 1, 2);
        conv2d.weights({ //1 1 2 2
            1.0f,2.0f,
            3.0f,4.0f,
        });
        conv2d.bias({0.0f});
        Tensor result = conv2d(input);
        std::cout << result << std::endl;
    }
    #endif

    #if 1
    {
        Tensor input(1, 2, 4, 4, {
            1.f, 2.f, 3.f, 4.f,
            5.f, 6.f, 7.f, 8.f,
            9.f, 10.f, 11.f, 12.f,
            13.f, 14.f, 15.f, 16.f,

            1.f, 2.f, 3.f, 4.f,
            5.f, 6.f, 7.f, 8.f,
            9.f, 10.f, 11.f, 12.f,
            13.f, 14.f, 15.f, 16.f,
        });
        //print_numpy(std::cout, input) << std::endl;

        Conv2d conv2d(2, 1, 3);
        conv2d.weights({ //1 2 3 3
            1.f, 2.f, 3.f,
            4.f, 5.f, 6.f,
            7.f, 8.f, 9.f,

            1.f, 2.f, 3.f,
            4.f, 5.f, 6.f,
            7.f, 8.f, 9.f,
        });
        conv2d.bias({0.0f});
        Tensor result = conv2d(input);
        std::cout << result << std::endl;
    }
    #endif
}

TEST_CASE("ConvTranspose2d")
{
    using namespace d2p2;
    #if 1
    {
        Tensor input(1, 1, 2, 2, {
            1.f, 2.f, 3.f, 4.f,
        });
        std::cout << "input:" << input << std::endl;

        ConvTranspose2d conv2d(1, 1, 2);
        conv2d.weights({ //1 1 2 2
            1.0f,2.0f,
            3.0f,4.0f,
        });
        conv2d.bias({0.0f});
        Tensor result = conv2d(input);
        std::cout << "resut: " << result << std::endl;
    }
    #endif

    #if 1
    {
        Tensor input(1, 3, 2, 2, {
            1.f, 2.f, 3.f, 4.f,
            1.f, 2.f, 3.f, 5.f,
            1.f, 2.f, 3.f, 6.f,
        });
        std::cout << "input:" << input << std::endl;

        ConvTranspose2d conv2d(3, 2, 2, 2);
        conv2d.weights({ //3 2 2 2
            1.0f,2.0f,
            3.0f,4.0f,
            5.0f,6.0f,
            7.0f,8.0f,
            9.0f,10.0f,
            11.0f,12.0f,
            13.0f,14.0f,
            15.0f,16.0f,
            17.0f,18.0f,
            19.0f,20.0f,
            21.0f,22.0f,
            23.0f,24.0f,
        });
        conv2d.bias({0.0f,0.0f});
        Tensor result = conv2d(input);
        std::cout << "resut: " << result << std::endl;
    }
    #endif
}

TEST_CASE("Linear")
{
    using namespace d2p2;
    #if 1
    {
        Tensor input(2, 3, 2, {
            1.f, 2.f, 3.f,
            4.f, 5.f, 6.f,
            7.f, 8.f, 9.f,
            10.f, 11.f, 12.f,
        });
        std::cout << input << std::endl;

        Linear linear(2,3);
        linear.weights({ //3 2
            1.f, 2.f, 3.f,
            4.f, 5.f, 6.f,
        });
        linear.bias({0.0f, 0.0f, 0.0f});
        Tensor result = linear(input);
        std::cout << result << std::endl;
    }
    #endif
}

