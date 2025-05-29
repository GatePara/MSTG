#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <algorithm>
#include <iomanip>
#include <filesystem>
#include <string>

struct Interval {
    int start;
    int end;
};

int count_hits(int q_start, int q_end, const std::vector<Interval>& base_ranges) {
    int count = 0;
    for (const auto& base : base_ranges) {
        if (!(base.end < q_start || base.start > q_end)) {
            ++count;
        }
    }
    return count;
}

int find_R_given_L(int L, double target_sel, const std::vector<Interval>& base_ranges, int total_base, int universe, double tolerance) {
    int left = L;
    int right = universe - 1;

    while (left <= right) {
        int mid = (left + right) / 2;
        int hits = count_hits(L, mid, base_ranges);
        double sel = static_cast<double>(hits) / total_base;

        if (std::abs(sel - target_sel) <= tolerance) {
            return mid;
        } else if (sel < target_sel) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }

    return -1;
}

void generate_queries(double sel, int queries_per_bucket, const std::vector<Interval>& base_ranges, int total_base, int universe, double tolerance, const std::string& output_dir) {
    std::vector<Interval> query_list;
    int total_hit = 0;
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> uni(0, universe - 1);

    std::cout << "\ngenerate sel = " << std::fixed << std::setprecision(2) << sel << " query\n";

    for (int i = 0; i < queries_per_bucket; ++i) {
        bool found = false;
        int attempts = 0;
        Interval latest_query;
        double real_sel = 0.0;

        while (!found && attempts < 10000) {
            int L = uni(rng);
            int R = find_R_given_L(L, sel, base_ranges, total_base, universe, tolerance);
            if (R == -1) {
                ++attempts;
                continue;
            }

            int hit = count_hits(L, R, base_ranges);
            real_sel = static_cast<double>(hit) / total_base;

            if (std::abs(real_sel - sel) <= tolerance) {
                found = true;
                query_list.push_back({L, R});
                total_hit += hit;
                latest_query = {L, R};
            }

            ++attempts;
        }

        if (!found) {
            std::cerr << "❌ generate, failed when sel = " << sel << "\n";
            exit(1);
        }

        std::cout << "[" << i + 1 << "/" << queries_per_bucket << "] "
                  << "L=" << latest_query.start << ", R=" << latest_query.end << ",sel=" << std::fixed << std::setprecision(4) << real_sel
                  << " ✅\r" << std::flush;
    }

    double avg_sel = static_cast<double>(total_hit) / (total_base * queries_per_bucket);
    double avg_len = 0.0;
    for (const auto& q : query_list) {
        avg_len += (q.end - q.start + 1);
    }
    avg_len /= query_list.size();

    std::string filename = output_dir + "/query_sel" + std::to_string(sel).substr(0, 4) + ".bin";
    std::ofstream out(filename, std::ios::binary);
    for (const auto& q : query_list) {
        out.write(reinterpret_cast<const char*>(&q.start), sizeof(int));
        out.write(reinterpret_cast<const char*>(&q.end), sizeof(int));
    }

}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <base_range_path> <universe_size>\n";
        return 1;
    }

    const std::string base_path = argv[1];
    const int universe = std::stoi(argv[2]);
    const std::string output_dir = "query_intervals_0_" + std::to_string(universe-1);

    std::filesystem::create_directories(output_dir);

    std::vector<double> target_sels = {0.05, 0.10, 0.20, 0.30, 0.40};
    const int queries_per_bucket = 1000;
    const double tolerance = 0.01;

    std::ifstream infile(base_path, std::ios::binary);
    if (!infile) {
        std::cerr << "Failed to open file: " << base_path << "\n";
        return 1;
    }

    std::vector<Interval> base_ranges;
    int s, e;
    while (infile.read(reinterpret_cast<char*>(&s), sizeof(int)) &&
           infile.read(reinterpret_cast<char*>(&e), sizeof(int))) {
        base_ranges.push_back({s, e});
    }
    int total_base = base_ranges.size();
    std::cout << "Loaded base intervals: " << total_base << "\n";

    for (double sel : target_sels) {
        generate_queries(sel, queries_per_bucket, base_ranges, total_base, universe, tolerance, output_dir);
    }

    return 0;
}