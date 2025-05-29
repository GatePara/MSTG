#include "mstg_final.h"
#include <iostream>
#include <stdexcept>
#include <string>
#include <chrono>
#include <iomanip>
#include <ctime>

std::string get_current_time_string()
{
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);

    std::tm tm_now;
#ifdef _WIN32
    localtime_s(&tm_now, &t); // Windows
#else
    localtime_r(&t, &tm_now); // Linux / Unix
#endif

    std::ostringstream oss;
    oss << std::put_time(&tm_now, "%Y-%m-%d %H:%M:%S");
    return oss.str();
}

struct BuildConfig
{
    std::string base_data_path;
    std::string base_range_path;
    std::string index_path;
    int M = 0;
    int ef_construction = 0;
    int threads = 0;

    void print() const
    {
        std::cout << "========= Build Config =========" << std::endl;
        std::cout << "Start time        : " << get_current_time_string() << std::endl;
        std::cout << "Data path         : " << base_data_path << std::endl;
        std::cout << "Range path        : " << base_range_path << std::endl;
        std::cout << "Index output path : " << index_path << std::endl;
        std::cout << "M                 : " << M << std::endl;
        std::cout << "ef_construction   : " << ef_construction << std::endl;
        std::cout << "Threads           : " << threads << std::endl;
        std::cout << "================================" << std::endl;
    }
};

BuildConfig parse_arguments(int argc, char **argv)
{
    BuildConfig config;
    for (int i = 1; i < argc; i++)
    {
        std::string arg = argv[i];
        if (arg == "--data_path" && i + 1 < argc)
        {
            config.base_data_path = argv[++i];
        }
        else if (arg == "--data_range_path" && i + 1 < argc)
        {
            config.base_range_path = argv[++i];
        }
        else if (arg == "--index_path" && i + 1 < argc)
        {
            config.index_path = argv[++i];
        }
        else if (arg == "--M" && i + 1 < argc)
        {
            config.M = std::stoi(argv[++i]);
        }
        else if (arg == "--ef_construction" && i + 1 < argc)
        {
            config.ef_construction = std::stoi(argv[++i]);
        }
        else if (arg == "--threads" && i + 1 < argc)
        {
            config.threads = std::stoi(argv[++i]);
        }
        else
        {
            throw std::runtime_error("Unknown or incomplete argument: " + arg);
        }
    }
    return config;
}

int main(int argc, char **argv)
{
    std::cout << "Executing build_intersection.cpp" << std::endl;
    try
    {
        BuildConfig config = parse_arguments(argc, argv);
        config.print();
        if (config.base_data_path.empty())
            throw std::runtime_error("Error: --data_path is missing or empty.");
        if (config.base_range_path.empty())
            throw std::runtime_error("Error: --data_range_path is missing or empty.");
        if (config.index_path.empty())
            throw std::runtime_error("Error: --index_path is missing or empty.");
        if (config.M <= 0)
            throw std::runtime_error("Error: --M should be a positive integer.");
        if (config.ef_construction <= 0)
            throw std::runtime_error("Error: --ef_construction should be a positive integer.");
        if (config.threads <= 0)
            throw std::runtime_error("Error: --threads should be a positive integer.");

         MultiSegmentTreeGraph::DataLoader storage;
        storage.LoadData(config.base_data_path);
        storage.LoadBaseRange(config.base_range_path);

         MultiSegmentTreeGraph::MSTGIndex<float> index(&storage, config.M, config.ef_construction, config.threads);

        auto start = std::chrono::steady_clock::now();

        index.Build();

        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "[Time] Build took " << elapsed.count() << " seconds." << std::endl;
        index.Save(config.index_path);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
