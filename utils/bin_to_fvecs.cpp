#include <iostream>
#include <fstream>
#include <cstring>
#include <vector>

int main(int argc, char **argv)
{
    if (argc != 4)
    {
        std::cout << "Usage: " << argv[0] << " <float/int8/uint8> input_bin output_fvecs" << std::endl;
        return -1;
    }

    int datasize = sizeof(float);
    if (strcmp(argv[1], "uint8") == 0 || strcmp(argv[1], "int8") == 0)
    {
        datasize = sizeof(uint8_t);
    }
    else if (strcmp(argv[1], "float") != 0)
    {
        std::cerr << "Unsupported type: use float/int8/uint8" << std::endl;
        return -1;
    }

    std::ifstream reader(argv[2], std::ios::binary);
    if (!reader)
    {
        std::cerr << "Error opening input file!" << std::endl;
        return -1;
    }

    int32_t npts = 0, dim = 0;
    reader.read((char *)&npts, sizeof(int32_t));
    reader.read((char *)&dim, sizeof(int32_t));
    std::cout << "Input BIN has " << npts << " points, each with dimension " << dim << std::endl;

    std::ofstream writer(argv[3], std::ios::binary);
    if (!writer)
    {
        std::cerr << "Error opening output file!" << std::endl;
        return -1;
    }

    size_t total_bytes = static_cast<size_t>(npts) * dim * datasize;
    std::vector<uint8_t> buf(total_bytes);
    reader.read((char *)buf.data(), total_bytes);

    for (int i = 0; i < npts; ++i)
    {
        writer.write((char *)&dim, sizeof(int32_t));
        writer.write((char *)(buf.data() + i * dim * datasize), dim * datasize);
    }

    reader.close();
    writer.close();
    std::cout << "Conversion complete!" << std::endl;
    return 0;
}
