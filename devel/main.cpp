#include <cstdint>
#include <cmath>
#include "src/ispc/matrix.ispc.h"
#include <iostream>
#include "d2p2/tensor.h"
#include "d2p2/util.h"

int main(void)
{
    std::vector<std::filesystem::path> train_files = d2p2::parse_directory("data/mnist_png/training",
        [](const std::filesystem::directory_entry& entry){
            std::filesystem::path ext = entry.path().extension();
            return ".png" == ext;
        }, false);

    for(size_t i=0; i<train_files.size(); ++i){
        std::cout << "[" << i << "] " << train_files[i] << std::endl;
    }
    return 0;
}
