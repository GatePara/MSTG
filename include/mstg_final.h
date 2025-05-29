#pragma once

#include <thread>
#include <map>
#include "utils.h"
#include "searcher.hpp"
#include <bitset>
#include <set>

namespace  MultiSegmentTreeGraph
{
    template <typename dist_t>
    class MSTGIndex
    {
    public:
        struct build_pruned_edge
        {
            float dist;
            int id;
            int start_version;
            int end_version;

            build_pruned_edge(float d, int i, int s, int e) : dist(d), id(i), start_version(s), end_version(e) {}

            bool operator>(const build_pruned_edge &other) const
            {
                return dist > other.dist;
            }
            bool operator<(const build_pruned_edge &other) const
            {
                return dist < other.dist;
            }
        };

        struct build_edge
        {
            float dist;
            int id;

            build_edge(float d, int i) : dist(d), id(i) {}

            bool operator>(const build_edge &other) const
            {
                return dist > other.dist;
            }
            bool operator<(const build_edge &other) const
            {
                return dist < other.dist;
            }
        };

        struct mstg_edge
        {
            int id;
            int start_version;
            int end_version;
            int base_r = 0;
            mstg_edge() : id(-1), start_version(-1), end_version(-1), base_r(0) {}
            mstg_edge(int i, int s, int e) : id(i), start_version(s), end_version(e) {}
            mstg_edge(int i, int s, int e, int r) : id(i), start_version(s), end_version(e), base_r(r) {}
        };

        struct legacy_mstg_edge
        {
            int id;
            int start_version;
            int end_version;
        };

        struct FlatMSTG
        {
            std::vector<mstg_edge, memory::align_alloc<mstg_edge>> edges;
            std::vector<std::pair<size_t, size_t>, memory::align_alloc<std::pair<size_t, size_t>>> offsets;
            int max_layer;
            int data_nb;
        };

        struct search_parameters
        {
            int edge_limit;
            std::array<std::array<int, 3>, 2> VL_RL_RR; // {version_L_limit,RL,RR}
            std::array<int, 2> versions{-1, -1};
            std::array<TreeNode *, 2> roots{nullptr, nullptr};
        };

        constexpr static size_t ALIGN_BYTES = 64;
        constexpr static size_t EDGE_SIZE = sizeof(mstg_edge);
        constexpr static size_t EDGE_PER_LINE = ALIGN_BYTES / EDGE_SIZE;

        using mstg_build = std::vector<std::vector<std::vector<build_edge>>>;
        using mstg_build_pruned = std::vector<std::vector<std::vector<build_pruned_edge>>>;

        // for build mstg
        MSTGIndex(DataLoader *store, int M_out = 16, int ef_c = 200, int threads = 16, bool range_filter = false) : storage(store), M(M_out), ef_construction(ef_c), max_threads(threads), range_filter(range_filter)
        {
            if (storage == nullptr)
                throw std::runtime_error("Error: dataLoader store shouldn't be a nullptr.");
            if (storage->base_memory_ == nullptr)
                throw std::runtime_error("Error: base_memory_ shouldn't be a nullptr.");
            vector_memory_ = storage->base_memory_;
            aligned_dim_ = storage->aligned_dim;
            size_data_per_aligned_vector = aligned_dim_ * sizeof(float);
            max_elements = storage->data_nb;
            space = new hnswlib::L2Space(aligned_dim_);
            fstdistfunc_ = space->get_dist_func();
            dist_func_param_ = space->get_dist_func_param();

            prefetch_lines = size_data_per_aligned_vector >> 6;

            std::unordered_set<int> unique_L;
            std::set<std::pair<int, int>> unique_LR;
            for (const auto &range : storage->base_range)
            {
                min_L = std::min(min_L, range.first);
                min_R = std::min(min_R, range.second);
                max_L = std::max(max_L, range.first);
                max_R = std::max(max_R, range.second);
                unique_L.insert(range.first);
                unique_LR.insert(range);
            }
            std::cout << "unique_L.size(): " << unique_L.size() << std::endl;
            std::cout << "unique_LR.size(): " << unique_LR.size() << std::endl;

            // when filter is range_filter, only reverse_tree is needed
            if (!range_filter)
            {
                forward_tree = new MultiSegmentTree(min_R, max_R, storage);
                init_edges(forward_tree, forward_edges, forward_edges_pruned);
            }

            reverse_tree = new MultiSegmentTree(min_R, max_R, storage);
            init_edges(reverse_tree, reverse_edges, reverse_edges_pruned);

            tree_nodes_.reserve(2 * max_elements * (reverse_tree->max_depth + 1));
            visited_list_pool_ = std::make_unique<hnswlib::VisitedListPool>(1, max_elements); // for parallel build
            log_every = max_elements / 10;
        }

        // for search on mstg
        MSTGIndex(DataLoader *store, std::string index_path_prefix, bool range_filter = false) : storage(store), range_filter(range_filter)
        {
            if (storage == nullptr)
                throw std::runtime_error("Error: dataLoader store shouldn't be a nullptr.");
            if (storage->base_memory_ == nullptr)
                throw std::runtime_error("Error: base_memory_ shouldn't be a nullptr.");

            vector_memory_ = storage->base_memory_;
            aligned_dim_ = storage->aligned_dim;
            size_data_per_aligned_vector = aligned_dim_ * sizeof(float);
            max_elements = storage->data_nb;
            space = new hnswlib::L2Space(aligned_dim_);
            fstdistfunc_ = space->get_dist_func();
            dist_func_param_ = space->get_dist_func_param();
            prefetch_lines = size_data_per_aligned_vector >> 6;

            std::unordered_set<int> unique_L;
            std::set<std::pair<int, int>> unique_LR;
            for (const auto &range : storage->base_range)
            {
                min_L = std::min(min_L, range.first);
                min_R = std::min(min_R, range.second);
                max_L = std::max(max_L, range.first);
                max_R = std::max(max_R, range.second);
                unique_L.insert(range.first);
                unique_LR.insert(range);
            }
            std::cout << "unique_L.size(): " << unique_L.size() << std::endl;
            std::cout << "unique_LR.size(): " << unique_LR.size() << std::endl;

            if (!range_filter)
            {
                forward_tree = new MultiSegmentTree(min_R, max_R, storage);
            }
            reverse_tree = new MultiSegmentTree(min_R, max_R, storage);

            tree_nodes_.reserve(max_elements * (reverse_tree->max_depth + 1));

            if (!range_filter)
            {
                std::vector<int> ascending_order(max_elements);
                std::iota(ascending_order.begin(), ascending_order.end(), 0);
                // sort id by L in ascending order
                std::stable_sort(ascending_order.begin(), ascending_order.end(), [&](int a, int b)
                                 {
                        if (storage->base_range[a].first != storage->base_range[b].first)
                            return storage->base_range[a].first < storage->base_range[b].first;
                        return storage->base_range[a].second < storage->base_range[b].second; });
                build_pst_by_sorted_l(ascending_order, forward_tree);
            }

            {
                std::vector<int> descending_order(max_elements);
                std::iota(descending_order.begin(), descending_order.end(), 0);
                // sort id by L in descending order
                std::stable_sort(descending_order.begin(), descending_order.end(), [&](int a, int b)
                                 {
                        if (storage->base_range[a].first != storage->base_range[b].first)
                            return storage->base_range[a].first > storage->base_range[b].first; //!
                        return storage->base_range[a].second < storage->base_range[b].second; });
                build_pst_by_sorted_l(descending_order, reverse_tree);
            }

            visited_list_pool_ = std::make_unique<hnswlib::VisitedListPool>(1, max_elements); // for parallel build
            log_every = max_elements / 10;

            std::string forward_index_save_path = index_path_prefix + ".forward";
            std::string reverse_index_save_path = index_path_prefix + ".reverse";
            if (!range_filter)
            {
                LoadEdgesToFlat(forward_index_save_path, forward_tree, forward_graph_flat);
            }
            LoadEdgesToFlat(reverse_index_save_path, reverse_tree, reverse_graph_flat);
        }

        void LoadEdgesToFlat(const std::string &save_path, MultiSegmentTree *tree, FlatMSTG &flat)
        {
            constexpr size_t ALIGN_BYTES = 64;
            constexpr size_t EDGE_SIZE = sizeof(mstg_edge);
            constexpr size_t EDGE_PER_LINE = ALIGN_BYTES / EDGE_SIZE;

            std::ifstream indexfile(save_path, std::ios::in | std::ios::binary);
            if (!indexfile.is_open())
                throw Exception("cannot open " + save_path);

            if (!storage)
                throw Exception("storage pointer is null");

            mstg_edge dummy = {-1, -1, -1, -1};

            uint64_t data_nb, total_bytes, max_depth;
            indexfile.read(reinterpret_cast<char *>(&data_nb), sizeof(uint64_t));
            indexfile.read(reinterpret_cast<char *>(&total_bytes), sizeof(uint64_t));
            indexfile.read(reinterpret_cast<char *>(&max_depth), sizeof(uint64_t));
            flat.data_nb = data_nb;
            flat.max_layer = max_depth;
            flat.edges.clear();
            uint64_t edge_count_estimate = total_bytes / sizeof(mstg_edge) + 1;
            flat.edges.reserve(edge_count_estimate);
            flat.offsets.clear();
            flat.offsets.reserve(data_nb * (max_depth + 1));
            for (int pid = 0; pid < data_nb; ++pid)
            {
                for (int layer = 0; layer <= max_depth; ++layer)
                {
                    int size = 0;
                    indexfile.read(reinterpret_cast<char *>(&size), sizeof(int));

                    if (layer == 0)
                    {
                        size_t cur_offset = flat.edges.size();
                        size_t misalign = cur_offset % EDGE_PER_LINE;
                        if (misalign != 0)
                        {
                            size_t pad = EDGE_PER_LINE - misalign;
                            flat.edges.insert(flat.edges.end(), pad, dummy);
                        }
                    }

                    struct legacy_mstg_edge
                    {
                        int id;
                        int start_version;
                        int end_version;
                    };
                    size_t start = flat.edges.size();
                    for (int i = 0; i < size; ++i)
                    {
                        int id;
                        int s;
                        int e;
                        indexfile.read(reinterpret_cast<char *>(&id), sizeof(int));
                        if (id & 0x80000000)
                        {
                            id &= 0x7FFFFFFF;
                            s = storage->base_range[id].first;
                            e = storage->base_range[id].first;
                        }
                        else
                        {
                            indexfile.read(reinterpret_cast<char *>(&s), sizeof(int));
                            indexfile.read(reinterpret_cast<char *>(&e), sizeof(int));
                        }
                        int r = storage->base_range[id].second;
                        mstg_edge edge{
                            id,
                            s,
                            e,
                            r};

                        flat.edges.push_back(edge);
                    }
                    size_t end = flat.edges.size();
                    flat.offsets.emplace_back(start, end);
                }
            }

            indexfile.close();
        }

        void SaveEdges(const std::string &save_path,
                       MultiSegmentTree *tree,
                       mstg_build &edges,
                       mstg_build_pruned &pruned_edges)
        {
            CheckPath(save_path);
            std::ofstream indexfile(save_path, std::ios::out | std::ios::binary);
            if (!indexfile.is_open())
                throw Exception("cannot open " + save_path);

            const uint64_t max_depth = tree->max_depth;
            const uint64_t data_nb = max_elements;

            uint64_t reserve = 0;
            for (uint64_t id = 0; id < data_nb; ++id)
            {
                uint64_t layer_bytes = 0;
                for (uint64_t layer = 0; layer <= max_depth; ++layer)
                {
                    layer_bytes += edges[id][layer].size() * 4 * sizeof(int) + pruned_edges[id][layer].size() * 4 * sizeof(int);
                }
                layer_bytes = (layer_bytes + 63) / 64 * 64;
                reserve += layer_bytes;
            }
            reserve = (reserve + 63) / 64 * 64;

            indexfile.write(reinterpret_cast<const char *>(&data_nb), sizeof(uint64_t));
            indexfile.write(reinterpret_cast<const char *>(&reserve), sizeof(uint64_t));
            indexfile.write(reinterpret_cast<const char *>(&max_depth), sizeof(uint64_t));

            std::vector<uint64_t> fwd_layer_total(max_depth + 1, 0);
            std::vector<uint64_t> rev_layer_total(max_depth + 1, 0);

            uint64_t fwd_global_total = 0;
            uint64_t rev_global_total = 0;

            for (uint64_t id = 0; id < data_nb; ++id)
            {
                std::vector<int> neighbor_ids;
                std::vector<build_pruned_edge> edge_cache;
                for (uint64_t layer = 0; layer <= max_depth; ++layer)
                {
                    neighbor_ids.clear();
                    edge_cache.clear();
                    for (auto &nbr : pruned_edges[id][layer])
                    {
                        edge_cache.emplace_back(nbr);
                        rev_layer_total[layer]++;
                        rev_global_total++;
                    }
                    for (auto &nbr : edges[id][layer])
                    {
                        build_pruned_edge cur = {nbr.dist, nbr.id, storage->base_range[nbr.id].first, storage->base_range[nbr.id].first};
                        edge_cache.emplace_back(cur);
                        fwd_layer_total[layer]++;
                        fwd_global_total++;
                    }
                    std::sort(edge_cache.begin(), edge_cache.end(), [](const build_pruned_edge &a, const build_pruned_edge &b)
                              { return a.dist < b.dist; });

                    int size = edge_cache.size();
                    indexfile.write(reinterpret_cast<const char *>(&size), sizeof(int));

                    for (build_pruned_edge &nbr : edge_cache)
                    {
                        if (nbr.start_version == nbr.end_version)
                        {
                            int id = nbr.id;
                            id |= 0x80000000;
                            indexfile.write(reinterpret_cast<const char *>(&id), sizeof(int));
                        }
                        else
                        {
                            indexfile.write(reinterpret_cast<const char *>(&nbr.id), sizeof(int));
                            indexfile.write(reinterpret_cast<const char *>(&nbr.start_version), sizeof(int));
                            indexfile.write(reinterpret_cast<const char *>(&nbr.end_version), sizeof(int));
                        }
                    }
                }
            }

            indexfile.close();

            // 打印分层统计信息
            std::cout << "==== Per-layer Edge Statistics ====\n";
            for (int layer = 0; layer <= max_depth; ++layer)
            {
                std::cout << "Layer " << layer
                          << " | FWD: " << fwd_layer_total[layer]
                          << " | REV: " << rev_layer_total[layer]
                          << " | TOTAL: " << (fwd_layer_total[layer] + rev_layer_total[layer])
                          << '\n';
            }

            std::cout << "\n==== Global Edge Statistics ====\n";
            std::cout << "FWD: " << fwd_global_total
                      << ", REV: " << rev_global_total
                      << ", TOTAL: " << (fwd_global_total + rev_global_total)
                      << "\n\n";
        }

        void Build()
        {
            if (has_built)
            {
                throw std::runtime_error("Index has been built, don't built again!");
            }

            std::cout << "sizeof(TreeNode): " << sizeof(TreeNode) << std::endl;
            std::cout << "min_L: " << min_L << " max_L: " << max_L << std::endl;
            std::cout << "min_R: " << min_R << " max_R: " << max_R << std::endl;

            // build forward index
            if (!range_filter)
            {
                std::cout << "[Log] Start build forward index." << std::endl;
                auto start = std::chrono::steady_clock::now();
                std::vector<int> order(max_elements);
                std::iota(order.begin(), order.end(), 0);
                // sort id by L in ascending order
                std::stable_sort(order.begin(), order.end(), [&](int a, int b)
                                 {
                        if (storage->base_range[a].first != storage->base_range[b].first)
                            return storage->base_range[a].first < storage->base_range[b].first;
                        return storage->base_range[a].second < storage->base_range[b].second; });
                build_mstg_by_sorted_l("forward", order, forward_tree, forward_edges, forward_edges_pruned);
                auto end = std::chrono::steady_clock::now();
                std::chrono::duration<double> elapsed = end - start;
                std::cout << "[Time] Build forward index took " << elapsed.count() << " seconds." << std::endl;
            }

            {
                std::cout << "[Log] Start build reverse index." << std::endl;
                auto start = std::chrono::steady_clock::now();
                std::vector<int> order(max_elements);
                std::iota(order.begin(), order.end(), 0);
                // sort id by L in descending order
                std::stable_sort(order.begin(), order.end(), [&](int a, int b)
                                 {
                        if (storage->base_range[a].first != storage->base_range[b].first)
                            return storage->base_range[a].first > storage->base_range[b].first; //! 
                        return storage->base_range[a].second < storage->base_range[b].second; });
                build_mstg_by_sorted_l("reverse", order, reverse_tree, reverse_edges, reverse_edges_pruned);
                auto end = std::chrono::steady_clock::now();
                std::chrono::duration<double> elapsed = end - start;
                std::cout << "[Time] Build reverse index took " << elapsed.count() << " seconds." << std::endl;
            }
            has_built = true;
        }

        void IntersectionSearch(std::vector<int> &SearchEF, std::ofstream &outfile, int edge_limit, int K, int repeats = 5)
        {
            if (range_filter)
            {
                throw std::runtime_error("can not search on range index");
            }
            outfile << "efSearch" << "," << "qps" << "," << "latency" << "," << "recall" << "," << "dco" << "," << "hops" << std::endl;
            std::vector<int> HOP;
            std::vector<int> DCO;
            std::vector<float> QPS;
            std::vector<float> RECALL;
            std::vector<float> LATENCY;
            std::vector<std::vector<int>> &gt = storage->groundtruth;
            if (gt.size() != storage->query_nb)
                throw std::runtime_error("gt.size()!=storage->query_nb");
            if (gt[0].size() < K)
                throw std::runtime_error("gt[0].size()<K");
            for (int repeat = 0; repeat < repeats; repeat++)
            {
                for (auto ef : SearchEF)
                {
                    size_t tp = 0;
                    metric_hops = 0;
                    metric_distance_computations = 0;
                    double total_ms = 0;
                    for (int i = 0; i < storage->query_nb; i++)
                    {
                        auto start = std::chrono::high_resolution_clock::now();
                        char *query_data = storage->GetQueryById(i);
                        int ql = storage->query_range[i].first, qr = storage->query_range[i].second;
                        std::priority_queue<PFI> knn;
                        std::priority_queue<PFI> merged;
                        std::unordered_set<int> seen;

                         MultiSegmentTreeGraph::MSTGIndex<float>::search_parameters params{};
                        std::vector<std::vector<int>> eps(2);
                        params.edge_limit = edge_limit;
                        params.VL_RL_RR = {{{ql, ql, max_R}, {qr, qr, max_R}}};
                        find_version_roots(params, eps);

                        bool r0 = (params.roots[0] != nullptr);
                        bool r1 = (params.roots[1] != nullptr);
                        switch ((r0 << 1) | r1)
                        {
                        case 0b11:
                            search_on_forward_pstg<true, true>(knn, query_data, ef, K, params, eps);
                            break;
                        case 0b10:
                            search_on_forward_pstg<true, false>(knn, query_data, ef, K, params, eps);
                            break;
                        case 0b01:
                            search_on_forward_pstg<false, true>(knn, query_data, ef, K, params, eps);
                            break;
                        default:
                            break;
                        }

                        merge_knn(knn, merged, seen, K);
                        search_on_reverse_pstg(knn, query_data, ef, K, ql, ql, qr, edge_limit);
                        merge_knn(knn, merged, seen, K);

                        auto end = std::chrono::high_resolution_clock::now();
                        auto duration_ms = std::chrono::duration<double, std::milli>(end - start).count();
                        total_ms += duration_ms;

                        std::map<int, int> record;
                        while (merged.size())
                        {
                            auto x = merged.top().second;
                            merged.pop();
                            if (record.count(x))
                                throw Exception("repetitive search results");
                            record[x] = 1;
                            if (std::find(gt[i].begin(), gt[i].begin() + K, x) != (gt[i].begin() + K))
                                tp++;
                        }
                    }

                    float recall = 1.0f * tp / (storage->query_nb * (size_t)K);
                    float dco = metric_distance_computations * 1.0 / storage->query_nb;
                    float hop = metric_hops * 1.0 / storage->query_nb;
                    float latency = total_ms / storage->query_nb;
                    float qps = storage->query_nb / (total_ms / 1000);
                    HOP.emplace_back(hop);
                    DCO.emplace_back(dco);
                    QPS.emplace_back(qps);
                    RECALL.emplace_back(recall);
                    LATENCY.emplace_back(latency);

                    outfile << ef << "," << QPS.back() << "," << LATENCY.back() << "," << RECALL.back() << "," << DCO.back() << "," << HOP.back() << std::endl;

                    if (recall > 0.999)
                    {
                        break;
                    }
                }
            }
        }

        void RangeSearch(std::vector<int> &SearchEF, std::ofstream &outfile, int edge_limit, int K, int repeats = 5)
        {
            if (!range_filter)
            {
                throw std::runtime_error("can not search on non-range index");
            }
            outfile << "efSearch" << "," << "qps" << "," << "latency" << "," << "recall" << "," << "dco" << "," << "hops" << std::endl;
            std::vector<int> HOP;
            std::vector<int> DCO;
            std::vector<float> QPS;
            std::vector<float> RECALL;
            std::vector<float> LATENCY;
            std::vector<std::vector<int>> &gt = storage->groundtruth;
            if (gt.size() != storage->query_nb)
                throw std::runtime_error("gt.size()!=storage->query_nb");
            if (gt[0].size() < K)
                throw std::runtime_error("gt[0].size()<K");
            for (int repeat = 0; repeat < repeats; repeat++)
            {
                for (auto ef : SearchEF)
                {
                    size_t tp = 0;
                    metric_hops = 0;
                    metric_distance_computations = 0;
                    double total_ms = 0;
                    for (int i = 0; i < storage->query_nb; i++)
                    {
                        auto start = std::chrono::high_resolution_clock::now();
                        char *query_data = storage->GetQueryById(i);
                        int ql = storage->query_range[i].first, qr = storage->query_range[i].second;
                        std::priority_queue<PFI> merged;
                        search_on_reverse_pstg(merged, query_data, ef, K, ql, ql, qr, edge_limit);
                        auto end = std::chrono::high_resolution_clock::now();
                        auto duration_ms = std::chrono::duration<double, std::milli>(end - start).count();
                        total_ms += duration_ms;

                        std::map<int, int> record;
                        while (merged.size())
                        {
                            auto x = merged.top().second;
                            merged.pop();
                            if (record.count(x))
                                throw Exception("repetitive search results");
                            record[x] = 1;
                            if (std::find(gt[i].begin(), gt[i].begin() + K, x) != (gt[i].begin() + K))
                                tp++;
                        }
                    }

                    float recall = 1.0f * tp / (storage->query_nb * (size_t)K);
                    float dco = metric_distance_computations * 1.0 / storage->query_nb;
                    float hop = metric_hops * 1.0 / storage->query_nb;
                    float latency = total_ms / storage->query_nb;
                    float qps = storage->query_nb / (total_ms / 1000);
                    HOP.emplace_back(hop);
                    DCO.emplace_back(dco);
                    QPS.emplace_back(qps);
                    RECALL.emplace_back(recall);
                    LATENCY.emplace_back(latency);

                    outfile << ef << "," << QPS.back() << "," << LATENCY.back() << "," << RECALL.back() << "," << DCO.back() << "," << HOP.back() << std::endl;

                    if (recall > 0.999)
                    {
                        break;
                    }
                }
            }
        }

        void Save(std::string save_index_prefix)
        {
            if (!has_built)
            {
                throw std::runtime_error("Index has not been built!");
            }
            std::string forward_index_save_path = save_index_prefix + ".forward";
            std::string reverse_index_save_path = save_index_prefix + ".reverse";

            if (!range_filter)
            {
                SaveEdges(forward_index_save_path, forward_tree, forward_edges, forward_edges_pruned);
            }
            SaveEdges(reverse_index_save_path, reverse_tree, reverse_edges, reverse_edges_pruned);
        }

        ~MSTGIndex()
        {

            for (auto *node : tree_nodes_)
            {
                if (node != nullptr)
                {
                    delete node;
                    node = nullptr;
                }
            }

            if (forward_tree)
            {
                if (forward_tree->root)
                {
                    delete forward_tree->root;
                    forward_tree->root = nullptr;
                }
                delete forward_tree;
                forward_tree = nullptr;
            }


            if (reverse_tree)
            {
                if (reverse_tree->root)
                {
                    delete reverse_tree->root;
                    reverse_tree->root = nullptr;
                }
                delete reverse_tree;
                reverse_tree = nullptr;
            }


            if (space)
            {
                delete space;
                space = nullptr;
            }
        }

    private:
        void merge_knn(std::priority_queue<PFI> &knn,
                       std::priority_queue<PFI> &merged,
                       std::unordered_set<int> &seen, int K)
        {
            while (!knn.empty())
            {
                auto t = knn.top();
                knn.pop();
                if (seen.insert(t.second).second)
                {
                    merged.push(t);
                }
            }

            while (merged.size() > K)
            {
                merged.pop();
            }
        }
        void build_pst_by_sorted_l(std::vector<int> &order, MultiSegmentTree *tree)
        {
            if (tree == nullptr)
                throw std::runtime_error("MultiSegmentTree shouldn't be a nullptr");
            if (order.size() != max_elements)
                throw std::runtime_error("order.size() shouldn't be equal to max_elements");
            if (tree->has_built)
                throw std::runtime_error("tree has built");

            int max_add_nodes = 0;
            std::vector<TreeNode *> nodes_cache;
            tree->new_node_count = 0;
            for (int i = 0; i < max_elements; i++)
            {
                nodes_cache.clear();
                int id = order[i];
                int new_node_count_old = tree->new_node_count;

                int l = storage->base_range[id].first;
                int r = storage->base_range[id].second;

                if (i == 0)
                    nodes_cache = tree->insert(l, r, 0, true, id);
                else
                    nodes_cache = tree->insert(l, r, 0, false, id);

                int add_nodes = tree->new_node_count - new_node_count_old;
                max_add_nodes = std::max(max_add_nodes, add_nodes);

                tree_nodes_.insert(tree_nodes_.end(), nodes_cache.begin(), nodes_cache.end());
            }
            tree->has_built = true;
        }

        void init_edges(MultiSegmentTree *tree, mstg_build &edges, mstg_build_pruned &pruned_edges)
        {
            if (edges.size() != max_elements)
            {
                edges.resize(max_elements);
            }
            if (pruned_edges.size() != max_elements)
            {
                pruned_edges.resize(max_elements);
            }

            int reserve_layers = 1;
            for (auto &layer : edges)
            {
                if (layer.size() != tree->max_depth + 1)
                {
                    layer.resize(tree->max_depth + 1);
                }

                for (int i = 0; i < layer.size(); i++)
                {
                    layer[i].clear();
                    if (i < reserve_layers)
                    {
                        layer[i].reserve(M);
                    }
                }
            }

            for (auto &layer : pruned_edges)
            {
                if (layer.size() != tree->max_depth + 1)
                {
                    layer.resize(tree->max_depth + 1);
                }

                for (int i = 0; i < layer.size(); i++)
                {
                    layer[i].clear();
                    if (i < reserve_layers)
                    {
                        layer[i].reserve(M);
                    }
                }
            }
        }

        void build_mstg_by_sorted_l(std::string task_info, std::vector<int> &order, MultiSegmentTree *tree, mstg_build &edges, mstg_build_pruned &pruned_edges)
        {
            if (tree == nullptr)
                throw std::runtime_error("MultiSegmentTree shouldn't be a nullptr");
            if (order.size() != max_elements)
                throw std::runtime_error("order.size() shouldn't be equal to max_elements");
            if (tree->has_built)
                throw std::runtime_error("tree has built");

            init_edges(tree, edges, pruned_edges);

            int max_add_nodes = 0;
            std::vector<TreeNode *> nodes_cache;
            tree->new_node_count = 0;
            for (int i = 0; i < max_elements; i++)
            {
                nodes_cache.clear();
                int id = order[i];
                int new_node_count_old = tree->new_node_count;

                int l = storage->base_range[id].first;
                int r = storage->base_range[id].second;

                if (i == 0)
                    nodes_cache = tree->insert(l, r, 0, true, id);
                else
                    nodes_cache = tree->insert(l, r, 0, false, id);

                int add_nodes = tree->new_node_count - new_node_count_old;
                max_add_nodes = std::max(max_add_nodes, add_nodes);

#pragma omp parallel for num_threads(max_threads)
                for (int j = 0; j < nodes_cache.size() - 1; j++)
                {
                    TreeNode *u = nodes_cache[j];
                    process_node_parallel(u, id, tree, edges, pruned_edges);
                }

                tree_nodes_.insert(tree_nodes_.end(), nodes_cache.begin(), nodes_cache.end());

                if (i % log_every == 0 || i + 1 == max_elements)
                {
                    if (i != 0)
                    {
                        int l = storage->base_range[id].first;
                        int r = storage->base_range[id].second;
                    }
                }
            }
            // SaveEdges(save_path, tree);
            tree->has_built = true;
        }

        void kcna_on_current_pstg(std::priority_queue<build_edge> &top_candidates, int depth, const char *query, int ef, std::vector<int> &enterpoints, int id, mstg_build &edges)
        {
            hnswlib::VisitedList *vl = visited_list_pool_->getFreeVisitedList();
            hnswlib::vl_type *visited_array = vl->mass;
            hnswlib::vl_type visited_array_tag = vl->curV;
            std::priority_queue<build_edge, std::vector<build_edge>, std::greater<build_edge>> candidate_set; // min heap

            for (auto ep : enterpoints)
            {
                if (ep == id)
                    continue;
                if (visited_array[ep] == visited_array_tag)
                    continue;
                visited_array[ep] = visited_array_tag;
                char *neighbor_data = getVectorById(ep);
                float dis = fstdistfunc_(query, neighbor_data, dist_func_param_);
                build_edge new_edge = {dis, ep};
                candidate_set.emplace(new_edge);
                top_candidates.emplace(new_edge);
            }

            float lowerBound = top_candidates.top().dist;

            // greedy search
            std::vector<int> filtered_edges;
            filtered_edges.reserve(M);
            while (!candidate_set.empty())
            {
                filtered_edges.clear();
                auto current_pair = candidate_set.top();
                if (current_pair.dist > lowerBound)
                    break;

                candidate_set.pop();
                int current_pointId = current_pair.id;

                size_t online_size = edges[current_pointId][depth].size();
                if (online_size != 0)
                {
                    _mm_prefetch((char *)(visited_array + edges[current_pointId][depth][0].id), _MM_HINT_T0);
                }

                for (size_t i = 0; i < online_size; i++)
                {
                    int neighborId = edges[current_pointId][depth][i].id;
                    if (i < online_size - 1)
                    {
                        _mm_prefetch((char *)(visited_array + edges[current_pointId][depth][i + 1].id), _MM_HINT_T0);
                    }
                    if (visited_array[neighborId] == visited_array_tag)
                        continue;
                    visited_array[neighborId] = visited_array_tag;
                    filtered_edges.push_back(neighborId);
                }

                size_t filtered_edges_size = filtered_edges.size();

                _mm_prefetch((char *)(getVectorById(filtered_edges[0])), _MM_HINT_T0);
                for (size_t i = 0; i < filtered_edges_size; i++)
                {
                    if (i < filtered_edges_size - 1)
                    {
                        _mm_prefetch((char *)(getVectorById(filtered_edges[i + 1])), _MM_HINT_T0);
                    }

                    int neighborId = filtered_edges[i];
                    char *neighbor_data = getVectorById(neighborId);
                    float dis = fstdistfunc_(query, neighbor_data, dist_func_param_);

                    if (top_candidates.size() < ef || dis < lowerBound)
                    {

                        build_edge new_edge = {dis, neighborId};
                        top_candidates.emplace(new_edge);
                        candidate_set.emplace(new_edge);

                        if (top_candidates.size() > ef)
                        {
                            top_candidates.pop();
                        }
                        if (top_candidates.size())
                        {
                            lowerBound = top_candidates.top().dist;
                        }
                    }
                }
            }

            while (top_candidates.size() > ef)
            {
                top_candidates.pop();
            }

            visited_list_pool_->releaseVisitedList(vl);
        }

        void rng_prune_after_search(std::priority_queue<build_edge> &search_result, int id, int depth)
        {

            if (search_result.size() < M)
            {
                return;
            }

            std::priority_queue<build_edge, std::vector<build_edge>, std::greater<build_edge>> queue_closest;
            std::vector<build_edge> edges_cache;

            while (search_result.size())
            {
                queue_closest.emplace(search_result.top());
                search_result.pop();
            }

            while (queue_closest.size())
            {

                if (edges_cache.size() >= M)
                    break;

                build_edge current_pair = queue_closest.top();
                int current_pair_id = current_pair.id;
                char *current_pair_data = getVectorById(current_pair_id);
                float dist_to_query = current_pair.dist;
                queue_closest.pop();

                bool good = true;

                for (build_edge &nbr : edges_cache)
                {
                    char *neighbor_data = getVectorById(nbr.id);
                    float dist = fstdistfunc_(neighbor_data, current_pair_data, dist_func_param_);
                    
                    if (dist < dist_to_query)
                    {
                        good = false;
                        break;
                    }
                }
                if (good)
                {
                    edges_cache.emplace_back(current_pair);
                }
            }

            for (auto &nbr : edges_cache)
            {
                search_result.emplace(nbr);
            }
        }

        void rng_prune_after_add_reverse(std::priority_queue<build_edge, std::vector<build_edge>, std::greater<build_edge>> &queue_closest, int id, int depth, int from_id, mstg_build &edges, mstg_build_pruned &pruned_edges)
        {

            if (queue_closest.size() < M)
            {
                return;
            }
            std::vector<build_edge> &online_edges_ref = edges[id][depth];
            std::vector<build_pruned_edge> &pruned_edges_ref = pruned_edges[id][depth];
            online_edges_ref.clear();

            while (queue_closest.size())
            {
        
                if (online_edges_ref.size() >= M)
                    break;

                build_edge current_pair = queue_closest.top();
                int current_pair_id = current_pair.id;
                char *current_pair_data = getVectorById(current_pair_id);
                float dist_to_query = current_pair.dist;
                queue_closest.pop();

                bool good = true;

                for (build_edge &nbr : online_edges_ref)
                {
                    char *nbr_data = getVectorById(nbr.id);
                    float dist = fstdistfunc_(nbr_data, current_pair_data, dist_func_param_); 
                    if (dist < dist_to_query)
                    {
                        good = false;
                        break;
                    }
                }
                if (good)
                {
                    online_edges_ref.emplace_back(current_pair);
                }
                else
                {
                    int start_version_L = storage->base_range[current_pair_id].first;
                    int end_version_L = storage->base_range[from_id].first;

                    build_pruned_edge cur = {dist_to_query, current_pair_id, start_version_L, end_version_L};


                    // Here we make a special case, because it is possible that two points have the same L and R, and are inserted into the same node consecutively,
                    // in which case the pruned edge has no meaning to be kept.
                    // Imagine if all points have the same L and R, then this index degenerates into a normal HNSW index, and having edges is enough,
                    // keeping these edges is meaningless.
                    if (start_version_L != end_version_L)
                    {
                        pruned_edges_ref.emplace_back(cur);
                    }
                }
            }
        }

        void add_edges(std::priority_queue<build_edge> &search_result, int id, int depth, mstg_build &edges, mstg_build_pruned &pruned_edges)
        {
            if (search_result.size() > M)
                throw std::runtime_error("Search result shouldn't be more than M candidates returned by RNG Prune");

            std::vector<build_edge> &online_edges_ref = edges[id][depth];
            online_edges_ref.clear();

            while (search_result.size() > 0)
            {
                online_edges_ref.emplace_back(search_result.top());
                search_result.pop();
            }

            std::vector<build_pruned_edge> &pruned_edges_ref = pruned_edges[id][depth];

            for (build_edge nbr : online_edges_ref)
            {
                int nbr_id = nbr.id;
                if (nbr_id == id)
                {
                    throw std::runtime_error("A element shouldn't be in its neighbor list");
                }

                nbr.id = id;
                std::vector<build_edge> &online_neighbor_edges_ref = edges[nbr_id][depth];

                if (online_neighbor_edges_ref.size() < M)
                {
                    online_neighbor_edges_ref.emplace_back(nbr);
                }
                else
                {
                    std::priority_queue<build_edge, std::vector<build_edge>, std::greater<build_edge>> queue_closest; // min heap
                    queue_closest.emplace(nbr);
                    for (auto &t : online_neighbor_edges_ref)
                        queue_closest.emplace(t);

                    rng_prune_after_add_reverse(queue_closest, nbr_id, depth, id, edges, pruned_edges);
                }
            }
        }

        void process_node_parallel(TreeNode *u, int id, MultiSegmentTree *tree, mstg_build &edges, mstg_build_pruned &pruned_edges)
        {

            if (u->vector_num <= 1)
            {
                return;
            }
            std::vector<int> enterpoints = tree->get_eps_as_vector(u);
            std::priority_queue<build_edge> search_result;
            kcna_on_current_pstg(search_result, u->depth, getVectorById(id), ef_construction, enterpoints, id, edges);
            rng_prune_after_search(search_result, id, u->depth);
            add_edges(search_result, id, u->depth, edges, pruned_edges);
        }

        void find_version_roots(search_parameters &sp, std::vector<std::vector<int>> &eps)
        {
            sp.versions[0] = forward_tree->find_version_lower(sp.VL_RL_RR[0][0]);
            sp.versions[1] = forward_tree->find_version_lower(sp.VL_RL_RR[1][0]);

            eps.resize(2);

            if (sp.versions[0] != -1)
            {
                eps[0] = forward_tree->get_filter_eps_from_root(forward_tree->version_roots[sp.versions[0]], sp.VL_RL_RR[0][1], sp.VL_RL_RR[0][2]);
            }
            if (sp.versions[1] != -1)
            {
                eps[1] = forward_tree->get_filter_eps_from_root(forward_tree->version_roots[sp.versions[1]], sp.VL_RL_RR[1][1], sp.VL_RL_RR[1][2]);
            }

            for (int i = 0; i < 2; i++)
            {
                if (sp.versions[i] == -1 || eps[i].size() == 0)
                {
                    sp.roots[i] = nullptr;
                }
                else
                {
                    sp.roots[i] = forward_tree->version_roots[sp.versions[i]];
                }
            }
        }

        template <bool R0, bool R1>
        void search_on_forward_pstg(std::priority_queue<PFI> &top_candidates, const char *query_data, int ef, int query_k, search_parameters &sp, std::vector<std::vector<int>> &eps)
        {

            hnswlib::VisitedList *vl = visited_list_pool_->getFreeVisitedList();
            hnswlib::vl_type *visited_array = vl->mass;
            hnswlib::vl_type visited_array_tag = vl->curV;
            std::priority_queue<PFI, std::vector<PFI>, std::greater<PFI>> candidate_set;

            for (int i = 0; i < 2; i++)
            {
                for (int j = 0; j < eps[i].size(); j++)
                {
                    int enterpoint = eps[i][j];
                    if (visited_array[enterpoint] == visited_array_tag)
                        continue;
                    visited_array[enterpoint] = visited_array_tag;
                    char *ep_data = getVectorById(enterpoint);
                    float dis = fstdistfunc_(query_data, ep_data, dist_func_param_);
                    ++metric_distance_computations;
                    candidate_set.emplace(dis, enterpoint);
                    top_candidates.emplace(dis, enterpoint);
                }
            }

            float lowerBound = top_candidates.top().first;
            std::vector<int> selected_edges;
            std::vector<int> filtered_edges;
            selected_edges.reserve(2 * sp.edge_limit + 1);
            filtered_edges.reserve(2 * sp.edge_limit + 1);
            while (!candidate_set.empty())
            {
                auto current_point_pair = candidate_set.top();
                float current_dist = current_point_pair.first;
                int current_id = current_point_pair.second;

                ++metric_hops;
                if (current_dist > lowerBound)
                {
                    break;
                }
                candidate_set.pop();
                fetch_forward_edges<R0, R1>(current_id, sp, vl, selected_edges);
                filtered_edges.clear();
                size_t num_edges = selected_edges.size();
                for (size_t i = 0; i < num_edges; ++i)
                {
                    _mm_prefetch((char *)(visited_array + selected_edges[i + 1]), _MM_HINT_T0);
                    int neighbor_id = selected_edges[i];
                    if (visited_array[neighbor_id] == visited_array_tag)
                    {
                        continue;
                    }
                    visited_array[neighbor_id] = visited_array_tag;
                    filtered_edges.push_back(selected_edges[i]);
                }

                for (size_t i = 0; i < filtered_edges.size(); ++i)
                {
                    _mm_prefetch((char *)(getVectorById(filtered_edges[i + 1])), _MM_HINT_T0);
                    int neighbor_id = filtered_edges[i];
                    char *neighbor_data = getVectorById(neighbor_id);
                    float dis = fstdistfunc_(query_data, neighbor_data, dist_func_param_);
                    ++metric_distance_computations;

                    if (top_candidates.size() < ef)
                    {
                        candidate_set.emplace(dis, neighbor_id);
                        top_candidates.emplace(dis, neighbor_id);
                        lowerBound = top_candidates.top().first;
                    }
                    else if (dis < lowerBound)
                    {
                        candidate_set.emplace(dis, neighbor_id);
                        top_candidates.emplace(dis, neighbor_id);
                        top_candidates.pop();
                        lowerBound = top_candidates.top().first;
                    }
                }
            }

            while (top_candidates.size() > query_k)
                top_candidates.pop();

            visited_list_pool_->releaseVisitedList(vl);
        }

        void search_on_reverse_pstg(std::priority_queue<PFI> &top_candidates, const char *query_data, int ef, int query_k, int version_L_limit, int RL, int RR, int edge_limit)
        {

            int version = -1;

            version = reverse_tree->find_version_higher(version_L_limit);

            if (version == -1)
            {
                return;
            }
            auto cur_root = reverse_tree->version_roots[version];
            std::vector<int> eps = reverse_tree->get_filter_eps_from_root(cur_root, RL, RR);
            if (eps.size() == 0)
            {
                return;
            }

            hnswlib::VisitedList *vl = visited_list_pool_->getFreeVisitedList();
            hnswlib::vl_type *visited_array = vl->mass;
            hnswlib::vl_type visited_array_tag = vl->curV;

            std::priority_queue<PFI, std::vector<PFI>, std::greater<PFI>> candidate_set;
            for (auto enterpoint : eps)
            {

                if (visited_array[enterpoint] == visited_array_tag)
                    continue;
                visited_array[enterpoint] = visited_array_tag;
                char *ep_data = getVectorById(enterpoint);
                float dis = fstdistfunc_(query_data, ep_data, dist_func_param_);
                ++metric_distance_computations;
                candidate_set.emplace(dis, enterpoint);
                top_candidates.emplace(dis, enterpoint);
            }

            float lowerBound = top_candidates.top().first;
            std::vector<tableint> selected_edges;
            selected_edges.reserve(edge_limit + 1);
            std::vector<tableint> filtered_edges;
            filtered_edges.reserve(edge_limit + 1);
            while (!candidate_set.empty())
            {
                auto current_point_pair = candidate_set.top();
                float current_dist = current_point_pair.first;
                int current_id = current_point_pair.second;
                ++metric_hops;
                if (current_dist > lowerBound)
                {
                    break;
                }
                candidate_set.pop();
                int base_R = storage->base_range[current_id].second;
                fetch_reverse_edges(current_id, base_R, RL, RR, edge_limit, visited_array, visited_array_tag, version_L_limit, selected_edges, cur_root);
                size_t num_edges = selected_edges.size();
                filtered_edges.clear();
                for (size_t i = 0; i < num_edges; ++i)
                {
                    _mm_prefetch((char *)(visited_array + selected_edges[i + 1]), _MM_HINT_T0);
                    int neighbor_id = selected_edges[i];
                    if (visited_array[neighbor_id] == visited_array_tag)
                    {
                        continue;
                    }
                    visited_array[neighbor_id] = visited_array_tag;
                    filtered_edges.push_back(selected_edges[i]);
                }

                for (size_t i = 0; i < filtered_edges.size(); ++i)
                {
                    _mm_prefetch((char *)(getVectorById(filtered_edges[i + 1])), _MM_HINT_T0);
                    int neighbor_id = filtered_edges[i];
                    char *neighbor_data = getVectorById(neighbor_id);
                    float dis = fstdistfunc_(query_data, neighbor_data, dist_func_param_);
                    ++metric_distance_computations;

                    if (top_candidates.size() < ef)
                    {
                        candidate_set.emplace(dis, neighbor_id);
                        top_candidates.emplace(dis, neighbor_id);
                        lowerBound = top_candidates.top().first;
                    }
                    else if (dis < lowerBound)
                    {
                        candidate_set.emplace(dis, neighbor_id);
                        top_candidates.emplace(dis, neighbor_id);
                        top_candidates.pop();
                        lowerBound = top_candidates.top().first;
                    }
                }
            }

            while (top_candidates.size() > query_k)
                top_candidates.pop();

            visited_list_pool_->releaseVisitedList(vl);
        }

        int get_overlap(int l, int r, int ql, int qr)
        {
            int L = std::max(l, ql);
            int R = std::min(r, qr);
            if (L > R)
                return -1; // no overlap
            return R - L + 1;
        }

        inline std::pair<const mstg_edge *, size_t> get_flat_edges(const FlatMSTG &G, int id, int layer)
        {
            size_t idx = id * (G.max_layer + 1) + layer;
            auto [start, end] = G.offsets[idx];
            return {G.edges.data() + start, end - start};
        }

        inline TreeNode *find_child_covering(TreeNode *node, int target)
        {
            if (!node)
                return nullptr;

            if (node->left_child &&
                node->left_child->lbound <= target &&
                node->left_child->rbound >= target)
            {
                return node->left_child;
            }
            if (node->right_child &&
                node->right_child->lbound <= target &&
                node->right_child->rbound >= target)
            {
                return node->right_child;
            }
            return nullptr;
        }

        template <bool R0, bool R1>
        void fetch_forward_edges(int id, search_parameters &sp,
                                 hnswlib::VisitedList *vl,
                                 std::vector<int> &selected_edges)
        {

            selected_edges.clear();
            hnswlib::vl_type *visited_array = vl->mass;
            hnswlib::vl_type visited_array_tag = vl->curV;
            int base_R = storage->base_range[id].second;
            int edge_limit = sp.edge_limit;

            if constexpr (R0)
            {
                int version_L_limit = sp.VL_RL_RR[0][0];
                int QR_L = sp.VL_RL_RR[0][1];
                int QR_R = sp.VL_RL_RR[0][2];
                TreeNode *cur_node = sp.roots[0];
                fetch_forward_edges_impl(id, cur_node, visited_array, visited_array_tag, version_L_limit, QR_L, QR_R, base_R, selected_edges, sp.edge_limit);
            }

            if constexpr (R1)
            {
                int version_L_limit = sp.VL_RL_RR[1][0];
                int QR_L = sp.VL_RL_RR[1][1];
                int QR_R = sp.VL_RL_RR[1][2];
                TreeNode *cur_node = sp.roots[1];
                fetch_forward_edges_impl(id, cur_node, visited_array, visited_array_tag, version_L_limit, QR_L, QR_R, base_R, selected_edges, sp.edge_limit);
            }
        }

        void fetch_reverse_edges(int id, int base_R, int QR_L, int QR_R, int edge_limit,
                                 hnswlib::vl_type *visited_array,
                                 hnswlib::vl_type visited_array_tag,
                                 int version_L_limit,
                                 std::vector<tableint> &selected_edges, TreeNode *cur_node)
        {
            selected_edges.clear();
            while (cur_node && !cur_node->child_empty())
            {
                TreeNode *next_node = nullptr;

                if (cur_node->left_child &&
                    cur_node->left_child->lbound <= base_R &&
                    cur_node->left_child->rbound >= base_R)
                {
                    next_node = cur_node->left_child;
                }
                else if (cur_node->right_child &&
                         cur_node->right_child->lbound <= base_R &&
                         cur_node->right_child->rbound >= base_R)
                {
                    next_node = cur_node->right_child;
                }
                else
                {
                    break;
                }
                int cur_overlap = get_overlap(cur_node->lbound, cur_node->rbound, QR_L, QR_R);
                int nxt_overlap = get_overlap(next_node->lbound, next_node->rbound, QR_L, QR_R);
                if (cur_overlap != nxt_overlap)
                {
                    break;
                }

                cur_node = next_node;
                if (cur_node->left_child)
                {
                    _mm_prefetch((char *)(cur_node->left_child), _MM_HINT_T0);
                }

                if (cur_node->right_child)
                {
                    _mm_prefetch((char *)(cur_node->right_child), _MM_HINT_T0);
                }
            }
            auto [edge_ptr, edge_cnt] = get_flat_edges(reverse_graph_flat, id, cur_node->depth);
            _mm_prefetch((char *)(visited_array + edge_ptr[0].id), _MM_HINT_T0);
            for (; cur_node;)
            {
                if (cur_node->left_child)
                {
                    _mm_prefetch((char *)(cur_node->left_child), _MM_HINT_T0);
                }

                if (cur_node->right_child)
                {
                    _mm_prefetch((char *)(cur_node->right_child), _MM_HINT_T0);
                }

                auto [edge_ptr, edge_cnt] = get_flat_edges(reverse_graph_flat, id, cur_node->depth);
                for (size_t i = 0; i < edge_cnt; ++i)
                {
                    const auto &e = edge_ptr[i];

                    bool valid = version_L_limit <= e.start_version &&
                                 (e.start_version == e.end_version || version_L_limit > e.end_version);
                    if (!valid)
                        continue;

                    if (e.base_r < QR_L || e.base_r > QR_R)
                        continue;

                    if (visited_array[e.id] == visited_array_tag)
                        continue;

                    selected_edges.emplace_back(e.id);
                    if (selected_edges.size() >= static_cast<size_t>(edge_limit))
                        return;
                }

                cur_node = find_child_covering(cur_node, base_R);
            }
        }

        void fetch_forward_edges_impl(int id, TreeNode *cur_node, hnswlib::vl_type *visited_array,
                                      hnswlib::vl_type visited_array_tag, int version_L_limit, int QR_L, int QR_R, int base_R, std::vector<int> &selected_edges, int edge_limit)
        {
            int edge_check = 0;
            while (cur_node && !cur_node->child_empty())
            {
                TreeNode *next_node = nullptr;

                if (cur_node->left_child &&
                    cur_node->left_child->lbound <= base_R &&
                    cur_node->left_child->rbound >= base_R)
                {
                    next_node = cur_node->left_child;
                }
                else if (cur_node->right_child &&
                         cur_node->right_child->lbound <= base_R &&
                         cur_node->right_child->rbound >= base_R)
                {
                    next_node = cur_node->right_child;
                }
                else
                {
                    break;
                }
                int cur_overlap = get_overlap(cur_node->lbound, cur_node->rbound, QR_L, QR_R);
                int nxt_overlap = get_overlap(next_node->lbound, next_node->rbound, QR_L, QR_R);
                if (cur_overlap != nxt_overlap)
                {
                    break;
                }

                cur_node = next_node;
                if (cur_node->left_child)
                {
                    _mm_prefetch((char *)(cur_node->left_child), _MM_HINT_T0);
                }
                if (cur_node->right_child)
                {
                    _mm_prefetch((char *)(cur_node->right_child), _MM_HINT_T0);
                }
            }
            auto [edge_ptr, edge_cnt] = get_flat_edges(forward_graph_flat, id, cur_node->depth);
            _mm_prefetch((char *)(visited_array + edge_ptr[0].id), _MM_HINT_T0);
            for (; cur_node;)
            {
                if (cur_node->left_child)
                {
                    _mm_prefetch((char *)(cur_node->left_child), _MM_HINT_T0);
                }

                if (cur_node->right_child)
                {
                    _mm_prefetch((char *)(cur_node->right_child), _MM_HINT_T0);
                }

                auto [edge_ptr, edge_cnt] = get_flat_edges(forward_graph_flat, id, cur_node->depth);
                for (size_t i = 0; i < edge_cnt; ++i)
                {
                    const auto &e = edge_ptr[i];

                    bool valid = version_L_limit >= e.start_version &&
                                 (e.start_version == e.end_version || version_L_limit < e.end_version);
                    if (!valid)
                        continue;

                    if (e.base_r < QR_L || e.base_r > QR_R)
                        continue;

                    if (visited_array[e.id] == visited_array_tag)
                        continue;

                    selected_edges.emplace_back(e.id);
                    edge_check++;
                    if (edge_check >= static_cast<size_t>(edge_limit))
                        return;
                }

                cur_node = find_child_covering(cur_node, base_R);
            }
        }

        inline char *getVectorById(int id) const
        {
            return vector_memory_ + id * size_data_per_aligned_vector;
        }

        size_t size_data_per_aligned_vector{0};
        char *vector_memory_{nullptr};
        size_t aligned_dim_{0};

        MultiSegmentTree *forward_tree{nullptr};
        MultiSegmentTree *reverse_tree{nullptr};
        size_t max_threads{16};
        DataLoader *storage{nullptr};
        size_t M{32}; // build_edge's max out degree
        size_t ef_construction{200};
        size_t max_elements{0};
        size_t log_every{1};
        bool range_filter{false};

        hnswlib::L2Space *space{nullptr};
        hnswlib::DISTFUNC<dist_t> fstdistfunc_{nullptr};
        void *dist_func_param_{nullptr};
        std::unique_ptr<hnswlib::VisitedListPool> visited_list_pool_{nullptr};

        FlatMSTG forward_graph_flat;
        FlatMSTG reverse_graph_flat;
        std::vector<TreeNode *> tree_nodes_;

        int min_L{std::numeric_limits<int>::max()};
        int max_L{std::numeric_limits<int>::min()};
        int min_R{std::numeric_limits<int>::max()};
        int max_R{std::numeric_limits<int>::min()};

        bool has_built = false;

        size_t metric_distance_computations{0};
        size_t metric_hops{0};
        int prefetch_lines{0};

        mstg_build forward_edges;
        mstg_build_pruned forward_edges_pruned;
        mstg_build reverse_edges;
        mstg_build_pruned reverse_edges_pruned;
    };
}