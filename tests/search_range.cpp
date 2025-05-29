#include "mstg_final.h"
std::unordered_map<std::string, std::string> paths;

const int query_K = 10;
int M;

void Generate( MultiSegmentTreeGraph::DataLoader &storage)
{
    storage.LoadData(paths["data_vector"]);
    storage.LoadBaseRange(paths["base_range_path"]);
     MultiSegmentTreeGraph::QueryGenerator generator(storage.data_nb, storage.query_nb, storage.min_R, storage.max_R, storage.base_range);
    std::cout << "data_nb = " << storage.data_nb << " query_nb = " << storage.query_nb << " min_R = " << storage.min_R << " max_R = " << storage.max_R << '\n';
    // generator.GenerateRange_Selectivity(paths["query_range_path"]);
    std::cout << "finish generate range " << '\n';
    storage.LoadQueryRange(paths["query_range_path"]);
    // std::cout << "begin generate groundtruth " << '\n';
    generator.GenerateGroundtruth(paths["groundtruth_path"], storage);
    // std::cout << "end generate groundtruth " << '\n';
}

void init()
{
    // data vectors should be sorted by the attribute values in ascending order
    paths["data_vector"] = "";

    paths["query_vector"] = "";
    // the path of document where query range files are saved
    paths["query_range_path"] = "";
    // the path of document where groundtruth files are saved
    paths["groundtruth_path"] = "";
    // the path where index file is saved
    paths["index"] = "";
    // the path of document where search result files are saved
    paths["result_path"] = "";
    // the path of document where base range fils are saved
    paths["base_range_path"] = "";
    // M is the maximum out-degree same as index build
    paths["M"] = "";
}

int main(int argc, char **argv)
{
    std::cout << "Executing search_range.cpp" << std::endl;

    for (int i = 0; i < argc; i++)
    {
        std::string arg = argv[i];
        if (arg == "--data_path")
            paths["data_vector"] = argv[i + 1];
        if (arg == "--query_path")
            paths["query_vector"] = argv[i + 1];
        if (arg == "--query_range_path")
            paths["query_range_path"] = argv[i + 1];
        if (arg == "--groundtruth_path")
            paths["groundtruth_path"] = argv[i + 1];
        if (arg == "--index_file")
            paths["index"] = argv[i + 1];
        if (arg == "--result_path")
            paths["result_path"] = argv[i + 1];
        if (arg == "--base_range_path")
            paths["base_range_path"] = argv[i + 1];
        if (arg == "--M")
            M = std::stoi(argv[i + 1]);
    }

    if (argc != 17)
        throw Exception("please check input parameters");
    std::cout << "begin start load data" << '\n';
     MultiSegmentTreeGraph::DataLoader storage;
    storage.LoadQuery(paths["query_vector"]);
    storage.query_K = query_K;
    // If it is the first run, Generate shall be called; otherwise, Generate can be skipped
    Generate(storage);
    std::cout << "end Generate " << '\n';
    storage.LoadQueryRange(paths["query_range_path"]);
    std::cout << "start load groundtruth " << '\n';
    storage.LoadGroundtruth(paths["groundtruth_path"]);
    std::cout << "end load groundtruth " << '\n';
    storage.LoadData(paths["data_vector"]);
    std::cout << "start init search " << '\n';
    std::cout << paths["data_vector"] << '\n';
    std::cout << paths["index"] << '\n';
    bool range_search = true;
     MultiSegmentTreeGraph::MSTGIndex<float> index(&storage, paths["index"], range_search);
    std::cout << "end init search " << '\n';
    // searchefs can be adjusted
    std::vector<int> SearchEF;
    if (storage.dim == 128)
    {
        SearchEF = {10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 70, 80, 90, 100, 120, 140, 160, 180, 250};
    }
    else if (storage.dim == 200)
    {
        SearchEF = {10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 70, 80, 90, 100, 120, 140, 160, 180, 250};
    }
    else if (storage.dim == 960)
    {
        SearchEF = {10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 70, 80, 90, 100, 115, 130, 145, 160, 180, 200, 230, 265, 300, 340, 380, 450, 600, 750};
    }
    else if (storage.dim == 2048)
    {
        SearchEF = {10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 70, 80, 90, 100, 120, 140, 170, 220, 280, 330, 400, 500, 600, 700, 800};
    }
    else
    {
       SearchEF = {
            10, 20, 30, 40, 50, 60, 70, 80,
            90, 100, 120, 140, 180, 220, 260, 320,
            380, 480, 600, 750, 950, 1150, 1350, 1600,2000,2500,3200,4000};
    }

    std::sort(SearchEF.begin(), SearchEF.end());
    std::vector<std::vector<int>> &gt = storage.groundtruth;
    size_t real_gt_num = 0;
    size_t real_num = 0;
    auto queryrange = storage.query_range;
    for (int i = 0; i < storage.query_nb; i++)
    {
        int ql = queryrange[i].first;
        int qr = queryrange[i].second;

        for (int j = 0; j < storage.data_nb; j++)
        {
            int dl = storage.base_range[j].first;
            int dr = storage.base_range[j].second;

            if (!(dr < ql || dl > qr)) // 相交
            {
                real_num++;
            }
        }
        real_gt_num += gt[i].size();
    }
    float select = (double)(real_num) / ((double)(storage.query_nb) * (double)storage.data_nb);
    std::cout << "real_gt_num: " << real_gt_num << std::endl;
    std::cout << "select ratio: " << select << std::endl;
    std::cout << "begin search " << '\n';

    std::string savepath = paths["result_path"] + get_current_timestamp() + ".csv";
    CheckPath(savepath);
    std::ofstream outfile(savepath);
    if (!outfile.is_open())
        throw Exception("cannot open " + savepath);

    index.RangeSearch(SearchEF, outfile, M, query_K, 3);
    outfile.close();
}