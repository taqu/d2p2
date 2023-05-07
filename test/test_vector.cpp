#include "catch_amalgamated.hpp"
#include "catch_wrap.hpp"
#include "d2p2/tensor.h"
#include "src/ispc/matrix.ispc.h"
#include <string>
#include <unordered_map>

TEST_CASE("VectorDot")
{
	{
        float A[] = {
            1, 2, 3, 4
        };
        float B[] = {
            5, 6, 7, 8
        };
        float r = ispc::vector_dot(4, A, B);
        EXPECT_FLOAT_EQ(70.0f, r);
    }
}

TEST_CASE("VectorDotBatch")
{
	{
        float A[] = {
            1, 2, 3, 4,
            5, 6, 7, 8
        };
        float B[] = {
            5, 6, 7, 8,
            9, 10, 11, 12
        };
        float r[2];
        ispc::vector_dot_batch(r, 2, 4, A, B);
        EXPECT_FLOAT_EQ(70.0f, r[0]);
        EXPECT_FLOAT_EQ(278.0f, r[1]);
    }
}
