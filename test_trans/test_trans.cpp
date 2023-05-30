#include <cstdint>
#include <cstdio>
#include <cassert>
#include <utility>

uint32_t transpose_src2dst_size(uint32_t size, uint32_t kernel_size, uint32_t stride, uint32_t padding)
{
    assert(0 < size);
    assert(0 < kernel_size);
    assert(0 < stride);
    //uint32_t pad = std::max(kernel_size / 2, padding);
    //pad = (0 != (0x01U&kernel_size))? pad*2 : pad;
    //uint32_t r = size + (stride - 1) * (size - 1) + pad;
    return size + (size-1)*(stride-1) - 2*padding + (kernel_size-1);
}

int main(void)
{
	for(uint32_t size=1; size<11; ++size){
		for(uint32_t kernel_size=1; kernel_size<6; ++kernel_size){
			for(uint32_t stride = 1; stride<3; ++stride){
                uint32_t out_size = transpose_src2dst_size(size, kernel_size, stride, 0);
                printf("in:%d, kernel:%d, stride:%d, out:%d\n", size, kernel_size, stride, out_size);
			}
		}
	}
	return 0;
}
