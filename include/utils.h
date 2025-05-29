#pragma once

#include "space_l2.h"
#include <filesystem>
#include <string>
#include <cstring>
#include <vector>
#include <fstream>
#include <sys/time.h>
#include <algorithm>
#include <map>
#include <omp.h>
#include "memory.hpp"

inline std::string get_current_timestamp()
{
    auto now = std::chrono::system_clock::now();
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    std::tm tm_struct;

#ifdef _WIN32
    localtime_s(&tm_struct, &now_c); // Windows
#else
    localtime_r(&now_c, &tm_struct); // Linux / Mac
#endif

    std::ostringstream oss;
    oss << std::put_time(&tm_struct, "_%Y%m%d_%H%M%S"); // _YYYYMMDD_HHMMSS
    return oss.str();
}

inline void load_fvecs(char *filename, float *&data, unsigned &num, unsigned &dim)
{ // load data with sift pattern
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open())
    {
        std::cerr << "Error: Unable to open file " << filename << " for reading" << std::endl;
        exit(-1);
    }
    in.read((char *)&dim, 4);
    in.seekg(0, std::ios::end);
    uint64_t dim_64 = dim;
    std::ios::pos_type ss = in.tellg();
    size_t fsize = (size_t)ss;
    uint64_t num_64 = (size_t)(fsize / (dim_64 + 1) / 4);
    num = num_64;
    data = new float[num_64 * dim_64];

    in.seekg(0, std::ios::beg);
    for (uint64_t i = 0; i < num_64; i++)
    {
        in.seekg(4, std::ios::cur);
        in.read((char *)(data + i * dim_64), dim_64 * 4);
    }
    in.close();
}

inline void save_result(char *filename, std::vector<std::vector<unsigned>> &results)
{
    std::ofstream out(filename, std::ios::binary | std::ios::out);
    if (!out.is_open())
    {
        std::cerr << "Error: Unable to open file " << filename << " for writing" << std::endl;
        exit(-1);
    }
    for (unsigned i = 0; i < results.size(); i++)
    {
        unsigned GK = (unsigned)results[i].size();
        out.write((char *)&GK, sizeof(unsigned));
        out.write((char *)results[i].data(), GK * sizeof(unsigned));
    }
    out.close();
}

class Exception : public std::runtime_error
{
public:
    Exception(const std::string &msg) : std::runtime_error(msg) {}
};

void CheckPath(std::string filename)
{
    std::filesystem::path pathObj(filename);
    std::filesystem::path dirPath = pathObj.parent_path();
    if (!std::filesystem::exists(dirPath))
    {
        try
        {
            if (std::filesystem::create_directories(dirPath))
            {
                std::cout << "Directory created: " << dirPath << std::endl;
            }
            else
            {
                std::cerr << "Failed to create directory: " << dirPath << std::endl;
            }
        }
        catch (std::filesystem::filesystem_error &e)
        {
            throw Exception(e.what());
        }
    }
}

namespace MultiSegmentTreeGraph
{
    typedef std::pair<float, int> PFI;
    typedef unsigned int tableint;
    typedef unsigned int linklistsizeint;

    class DataLoader
    {
    public:
        int query_K;
        int min_L{std::numeric_limits<int>::max()};
        int max_L{std::numeric_limits<int>::min()};
        int min_R{std::numeric_limits<int>::max()};
        int max_R{std::numeric_limits<int>::min()};
        int data_nb{0}, query_nb{0};
        char *base_memory_{nullptr};
        char *query_memory_{nullptr};
        size_t query_dim, query_aligned_dim;
        int dim{0};
        size_t aligned_dim;
        size_t vector_bytes, query_vector_bytes;
        std::vector<std::pair<int, int>> query_range;
        std::vector<std::vector<int>> groundtruth;
        std::vector<std::pair<int, int>> base_range;

        DataLoader() {}
        ~DataLoader()
        {
            if (base_memory_ != nullptr)
            {
                free(base_memory_);
                base_memory_ = nullptr;
            }

            if (query_memory_ != nullptr)
            {
                free(query_memory_);
                query_memory_ = nullptr;
            }
        }

        inline char *GetBaseById(tableint id) const
        {
            return base_memory_ + id * vector_bytes;
        }

        inline char *GetQueryById(tableint id) const
        {
            return query_memory_ + id * query_vector_bytes;
        }

        void LoadQuery(std::string filename) // fbin format
        {
            std::ifstream infile(filename, std::ios::in | std::ios::binary);
            if (!infile.is_open())
                throw std::runtime_error("Cannot open file: " + filename);

            infile.read((char *)&query_nb, sizeof(int));
            infile.read((char *)&dim, sizeof(int));

            query_aligned_dim = ((dim + 7) / 8) * 8;
            query_vector_bytes = query_aligned_dim * sizeof(float);

            query_memory_ = (char *)memory::align_mm<1 << 21>(query_nb * query_vector_bytes);

            for (size_t i = 0; i < query_nb; i++)
            {
                infile.read(query_memory_ + i * query_vector_bytes, dim * sizeof(float));
            }

            infile.close();
        }

        void LoadData(std::string filename) // fbin format
        {
            std::ifstream infile(filename, std::ios::in | std::ios::binary);
            if (!infile.is_open())
                throw Exception("cannot open " + filename);
            infile.read((char *)&data_nb, sizeof(int));
            infile.read((char *)&dim, sizeof(int));
            std::cout << "data_nb: " << data_nb << " dim: " << dim << std::endl;

            aligned_dim = ((dim + 7) / 8) * 8;
            vector_bytes = aligned_dim * sizeof(float);

            base_memory_ = (char *)memory::align_mm<1 << 21>(data_nb * vector_bytes);
            for (size_t i = 0; i < (size_t)data_nb; i++)
            {
                infile.read(base_memory_ + i * vector_bytes, dim * sizeof(float));
            }

            infile.close();
        }

        void LoadBaseRange(std::string filename)
        {
            std::ifstream infile(filename, std::ios::in | std::ios::binary);
            if (!infile.is_open())
                throw Exception("Cannot open file: " + filename);

            infile.seekg(0, std::ios::end);
            std::streamsize file_size = infile.tellg();
            infile.seekg(0, std::ios::beg);
            // Calculate the expected file size based on the number of data entries
            std::streamsize expected_size = data_nb * 2 * sizeof(int);
            if (file_size < expected_size)
                throw Exception("File size mismatch: expected at least " + std::to_string(expected_size) +
                                " bytes, but got " + std::to_string(file_size) + " bytes.");

            // Read data entries
            for (int i = 0; i < data_nb; i++)
            {
                int bl, br;
                infile.read(reinterpret_cast<char *>(&bl), sizeof(int));
                infile.read(reinterpret_cast<char *>(&br), sizeof(int));
                min_L = std::min(bl, min_L);
                max_L = std::max(bl, max_L);
                min_R = std::min(br, min_R);
                max_R = std::max(br, max_R);
                base_range.emplace_back(bl, br);
            }

            infile.close();
        }

        void LoadGroundtruth(std::string filename)
        {
            size_t gt_size = 0;
            std::ifstream infile(filename, std::ios::in | std::ios::binary);
            if (!infile.is_open())
                throw Exception("cannot open " + filename);
            groundtruth.resize(query_nb);
            for (int i = 0; i < query_nb; i++)
            {
                int size = 0;
                infile.read((char *)&size, sizeof(int));

                groundtruth[i].resize(size);
                infile.read((char *)groundtruth[i].data(), size * sizeof(int));
                gt_size += size;
            }
            infile.close();
            std::cout << "loaded gt real num = " << gt_size << std::endl;
        }

        void LoadQueryRange(std::string filename)
        {
            std::cout << filename << '\n';
            std::ifstream infile(filename, std::ios::in | std::ios::binary);
            if (!infile.is_open())
                throw Exception("Cannot open file: " + filename);

            infile.seekg(0, std::ios::end);
            std::streamsize file_size = infile.tellg();
            infile.seekg(0, std::ios::beg);
            // Calculate the expected file size based on the number of data entries
            std::streamsize expected_size = query_nb * 2 * sizeof(int);
            if (file_size < expected_size)
                throw Exception("File size mismatch: expected at least " + std::to_string(expected_size) +
                                " bytes, but got " + std::to_string(file_size) + " bytes.");

            // Read data entries
            for (int i = 0; i < query_nb; i++)
            {
                int bl, br;
                infile.read(reinterpret_cast<char *>(&bl), sizeof(int));
                infile.read(reinterpret_cast<char *>(&br), sizeof(int));
                query_range.emplace_back(bl, br);
            }

            infile.close();
        }
    };

    class QueryGenerator
    {
    public:
        int data_nb, query_nb;
        int min_R, max_R;
        hnswlib::L2Space *space;
        hnswlib::DISTFUNC<float> fstdistfunc_{nullptr};
        void *dist_func_param_{nullptr};
        int range_size{1000};
        std::vector<std::pair<int, int>> base_range;

        QueryGenerator(int data_num, int query_num, int min_r, int max_r, std::vector<std::pair<int, int>> base_range) : data_nb(data_num), query_nb(query_num), min_R(min_r), max_R(max_r), base_range(base_range) {}
        ~QueryGenerator()
        {
            if (space != nullptr)
            {
                space = nullptr;
                delete space;
            }
        }

        float dis_compute(std::vector<float> &v1, std::vector<float> &v2)
        {
            hnswlib::DISTFUNC<float> fstdistfunc_ = space->get_dist_func();
            float dis = fstdistfunc_((char *)v1.data(), (char *)v2.data(), space->get_dist_func_param());
            return dis;
        }

        bool HasOverlap(int l, int r, int ql, int qr)
        {
            return l <= qr && ql <= r;
        }

        void GenerateGroundtruth(std::string savepath, DataLoader &storage)
        {
            space = new hnswlib::L2Space(storage.aligned_dim);
            fstdistfunc_ = space->get_dist_func();
            dist_func_param_ = space->get_dist_func_param();
            CheckPath(savepath);
            std::ofstream outfile(savepath, std::ios::out | std::ios::binary);
            if (!outfile.is_open())
                throw Exception("cannot open " + savepath);
            std::vector<std::vector<int>> gt;
            gt.resize(query_nb);
#pragma omp parallel for
            for (int i = 0; i < query_nb; i++)
            {
                char *query = storage.GetQueryById(i);
                auto t = storage.query_range;
                std::pair<int, int> rp = t[i];
                int ql = rp.first, qr = rp.second;
                std::priority_queue<std::pair<float, int>> ans;
                for (int j = 0; j < storage.data_nb; j++)
                {
                    int l = storage.base_range[j].first;
                    int r = storage.base_range[j].second;
                    if (!HasOverlap(l, r, ql, qr)) 
                        continue;
                    char *base = storage.GetBaseById(j);
                    float dis = fstdistfunc_(query, base, dist_func_param_);
                    ans.emplace(dis, j);

                    if (ans.size() > storage.query_K)
                        ans.pop();
                }

                int size = ans.size();
                gt[i].reserve(storage.query_K);
                while (ans.size())
                {
                    auto id = ans.top().second;
                    ans.pop();
                    gt[i].push_back(id);
                }
            }

            for (int i = 0; i < query_nb; i++)
            {
                int size = gt[i].size();
                if (size < storage.query_K)
                {
                    throw std::runtime_error("size < storage.query_K");
                }
                outfile.write((char *)&size, sizeof(int));
                outfile.write((char *)gt[i].data(), size * sizeof(int));
            }
            outfile.close();
        }
    };

    class TreeNode
    {
    public:
        int depth;
        int lbound, rbound;
        int vector_num{0};
        TreeNode *left_child{nullptr};
        TreeNode *right_child{nullptr};

        TreeNode(int l, int r, int d, TreeNode *lchild = nullptr, TreeNode *rchild = nullptr) : lbound(l), rbound(r), depth(d), left_child(lchild), right_child(rchild) {}

        void add_point(std::map<std::pair<int, int>, std::vector<int>> &vector_id_map, int id)
        {
            vector_id_map[{lbound, rbound}].push_back(id);
            this->vector_num++;
        }

        void add_point(std::map<std::pair<int, int>, std::vector<int>> &vector_id_map, int id, TreeNode *prev)
        {
            vector_id_map[{lbound, rbound}].push_back(id);
            this->vector_num = prev->vector_num + 1;
        }

        bool child_empty()
        {
            if (left_child != nullptr || right_child != nullptr)
            {
                return false;
            }
            return true;
        }
    };

    class MultiSegmentTree
    {
    public:
        DataLoader *storage;
        std::vector<TreeNode *> version_roots;
        std::vector<int> version_range_l;     
        TreeNode *root{nullptr};
        int max_depth{-1}; // 最大深度
        size_t new_node_count{0};
        std::mt19937 rng{42};
        // std::uniform_real_distribution<float> dist;
        std::map<std::pair<int, int>, std::vector<int>> vector_id_map;
        bool has_built{false};
        constexpr static int MAX_EP_NUM = 4;

        MultiSegmentTree(int min_R, int max_R, DataLoader *storage) : storage(storage)
        {
            root = new TreeNode(min_R, max_R, 0);
            max_depth = std::ceil(std::log2(max_R - min_R + 1));
        }

        std::vector<int> get_eps_as_vector(TreeNode *u)
        {

            std::vector<int> &node_vector_id = vector_id_map[{u->lbound, u->rbound}];
            if (u->vector_num <= MAX_EP_NUM)
            {
                return std::vector<int>(node_vector_id.begin(), node_vector_id.begin() + u->vector_num);
            }

            std::vector<int> eps;
            eps.reserve(MAX_EP_NUM);
            std::unordered_set<int> selected;
            selected.reserve(MAX_EP_NUM);
            std::uniform_int_distribution<> dis(0, u->vector_num - 1);

            while (eps.size() < MAX_EP_NUM)
            {
                int idx = dis(this->rng);
                if (selected.insert(idx).second)
                {
                    eps.push_back(node_vector_id[idx]);
                }
            }

            return eps;
        }

        int get_ep(TreeNode *u)
        {
            if (u->vector_num <= 1)
            {
                return vector_id_map[{u->lbound, u->rbound}].front();
            }
            else
            {
                std::vector<int> &node_vector_id = vector_id_map[{u->lbound, u->rbound}];
                std::uniform_int_distribution<> dis(0, u->vector_num - 1);
                int idx = dis(this->rng);
                return node_vector_id[idx];
            }
        }

        std::vector<TreeNode *> get_version_roots()
        {
            return version_roots;
        }

        std::vector<TreeNode *> insert(int l, int r, int depth, bool first_node, int id)
        {
            TreeNode *new_node = nullptr;
            std::vector<TreeNode *> res;
            res.clear();
            int L = root->lbound, R = root->rbound;

            if (first_node)
                new_node = InsertFirst(L, R, res, r, 0, id);
            else
                new_node = Insert(version_roots.back(), res, r, 0, id);

            version_roots.emplace_back(new_node);
            version_range_l.emplace_back(l);
            return res;
        }

        TreeNode *InsertFirst(int L, int R, std::vector<TreeNode *> &res, int value, int depth, int id)
        {
            TreeNode *new_node = new TreeNode(L, R, depth);
            new_node->add_point(vector_id_map, id);
            new_node_count++;

            res.emplace_back(new_node);

            if (L == R)
                return new_node;

            int mid = (L + R) >> 1;
            TreeNode *childnode = nullptr;
            if (value >= L && value <= mid)
            {
                childnode = InsertFirst(L, mid, res, value, depth + 1, id);
                new_node->left_child = childnode;
            }
            else
            {
                childnode = InsertFirst(mid + 1, R, res, value, depth + 1, id);
                new_node->right_child = childnode;
            }
            return new_node;
        }

        TreeNode *Insert(TreeNode *prev, std::vector<TreeNode *> &res, int value, int depth, int id)
        {
            int L = prev->lbound, R = prev->rbound;
            TreeNode *new_node = new TreeNode(L, R, depth);
            new_node->add_point(vector_id_map, id, prev);
            new_node_count++;

            res.emplace_back(new_node);

            if (L == R)
                return new_node;

            int mid = (L + R) >> 1;
            TreeNode *childnode = nullptr;
            if (value >= L && value <= mid)
            {
                if (prev->left_child != nullptr)
                {
                    auto prev_child = prev->left_child;
                    childnode = Insert(prev_child, res, value, depth + 1, id);
                    new_node->left_child = childnode;
                    new_node->right_child = prev->right_child;
                }
                else
                {
                    childnode = InsertFirst(L, mid, res, value, depth + 1, id);
                    new_node->left_child = childnode;
                    new_node->right_child = prev->right_child;
                }
            }
            else
            {
                if (prev->right_child != nullptr)
                {
                    auto prev_child = prev->right_child;
                    childnode = Insert(prev_child, res, value, depth + 1, id);
                    new_node->right_child = childnode;
                    new_node->left_child = prev->left_child;
                }
                else
                {
                    childnode = InsertFirst(mid + 1, R, res, value, depth + 1, id);
                    new_node->right_child = childnode;
                    new_node->left_child = prev->left_child;
                }
            }

            return new_node;
        }

        bool check_lower(int mid, int value)
        {
            // std::cout << "mid = " << mid << " version_range_l[mid] = " << version_range_l[mid] << '\n';
            if (version_range_l[mid] <= value)
                return true;
            else
                return false;
        }

        bool check_higher(int mid, int value)
        {
            // std::cout << "mid = " << mid << " version_range_l[mid] = " << version_range_l[mid] << '\n';
            if (version_range_l[mid] >= value)
                return true;
            else
                return false;
        }

        int find_version_lower(int value)
        {
            int id = -1;
            int l = 0, r = version_roots.size() - 1;
            while (l < r)
            {
                int mid = l + r >> 1;
                if (check_lower(mid, value))
                    id = mid, l = mid + 1;
                else
                    r = mid - 1;
            }
            return id;
        }

        int find_version_higher(int value)
        {
            int id = -1;
            int l = 0, r = version_roots.size() - 1;
            while (l < r)
            {
                int mid = l + r >> 1;
                if (check_higher(mid, value))
                    id = mid, l = mid + 1;
                else
                    r = mid - 1;
            }
            return id;
        }

        std::vector<TreeNode *> range_filter(TreeNode *u, int ql, int qr)
        {
            std::vector<TreeNode *> res;

            // 剪枝：当前节点区间完全不相交
            if (u == nullptr || u->rbound < ql || u->lbound > qr)
                return res;

            // 当前节点完全在查询区间内
            if (u->lbound >= ql && u->rbound <= qr)
                return {u};

            // 否则递归左右子树
            if (u->left_child)
            {
                auto left_res = range_filter(u->left_child, ql, qr);
                res.insert(res.end(), left_res.begin(), left_res.end());
            }

            if (u->right_child)
            {
                auto right_res = range_filter(u->right_child, ql, qr);
                res.insert(res.end(), right_res.begin(), right_res.end());
            }

            return res;
        }

        std::vector<int> get_filter_eps_from_root(TreeNode *u, int RL, int RR)
        {
            std::vector<int> res;

            if (u == nullptr || u->rbound < RL || u->lbound > RR)
                return res;

            if (u->lbound >= RL && u->rbound <= RR)
            {
                int ep = get_ep(u);
                res.push_back(ep);
                return res;
            }

            if (u->left_child)
            {
                auto left_res = get_filter_eps_from_root(u->left_child, RL, RR);
                res.insert(res.end(), left_res.begin(), left_res.end());
            }

            if (u->right_child)
            {
                auto right_res = get_filter_eps_from_root(u->right_child, RL, RR);
                res.insert(res.end(), right_res.begin(), right_res.end());
            }

            return res;
        }
    };
}