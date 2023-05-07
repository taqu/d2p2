#include "catch_amalgamated.hpp"
#include "catch_wrap.hpp"
#include "d2p2/tensor.h"
#include "src/ispc/matrix.ispc.h"
#include <string>
#include <unordered_map>

TEST_CASE("NN")
{
#if 0
    using namespace d2p2;
    std::unordered_map<std::string, Tensor> network;
    {
        network["W1"] = std::move(Tensor(2, 3, std::initializer_list<float>{0.1f, 0.3f, 0.5f, 0.2f, 0.4f, 0.6f}));
        network["b1"] = std::move(Tensor(1, 3, std::initializer_list<float>{0.1f, 0.2f, 0.3f}));
        network["W2"] = std::move(Tensor(3, 2, std::initializer_list<float>{0.1f, 0.4f, 0.2f, 0.5f, 0.3f, 0.6f}));
        network["b2"] = std::move(Tensor(1, 2, std::initializer_list<float>{0.1f, 0.2f}));
        network["W3"] = std::move(Tensor(2, 2, std::initializer_list<float>{0.1f, 0.3f, 0.2f, 0.4f}));
        network["b3"] = std::move(Tensor(1, 2, std::initializer_list<float>{0.1f, 0.2f}));
    }
    {
        Tensor& W1 = network["W1"];
        Tensor& b1 = network["b1"];
        Tensor& W2 = network["W2"];
        Tensor& b2 = network["b2"];
        Tensor& W3 = network["W3"];
        Tensor& b3 = network["b3"];

        Tensor x(1,2, {1.0f, 0.5f});
        Tensor a1 = x * W1 + b1;
        Tensor at = b2 + b3;
        std::cout << at << std::endl;
        Tensor z1 = sigmoid(a1);
        Tensor a2 = z1 * W2 + b2;
        Tensor z2 = sigmoid(a2);
        Tensor a3 = z2 * W3 + b3;
        Tensor y = identity(a3);
        EXPECT_FLOAT_EQ(0.31682708f, y(0,0));
        EXPECT_FLOAT_EQ(0.69627909f, y(0,1));
    }
#endif
}
