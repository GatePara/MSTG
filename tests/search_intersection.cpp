#include "mstg_final.h"
std::unordered_map<std::string, std::string> paths;

const int query_K = 10;
int M;

void Generate( MultiSegmentTreeGraph::DataLoader &storage)
{
    storage.LoadData(paths["data_vector"]);
    storage.LoadBaseRange(paths["baserange_saveprefix"]);
     MultiSegmentTreeGraph::QueryGenerator generator(storage.data_nb, storage.query_nb, storage.min_R, storage.max_R, storage.base_range);
    std::cout << "data_nb = " << storage.data_nb << " query_nb = " << storage.query_nb << " min_R = " << storage.min_R << " max_R = " << storage.max_R << '\n';
    // generator.GenerateRange_Selectivity(paths["range_saveprefix"]);
    std::cout << "finish generate range " << '\n';
    storage.LoadQueryRange(paths["range_saveprefix"]);
    std::cout << "begin generate groundtruth " << '\n';
    generator.GenerateGroundtruth(paths["groundtruth_saveprefix"], storage);
    std::cout << "end generate groundtruth " << '\n';
}

void init()
{
    // data vectors should be sorted by the attribute values in ascending order
    paths["data_vector"] = "";

    paths["query_vector"] = "";
    // the path of document where query range files are saved
    paths["range_saveprefix"] = "";
    // the path of document where groundtruth files are saved
    paths["groundtruth_saveprefix"] = "";
    // the path where index file is saved
    paths["index"] = "";
    // the path of document where search result files are saved
    paths["result_saveprefix"] = "";
    // the path of document where base range fils are saved
    paths["baserange_saveprefix"] = "";
    // M is the maximum out-degree same as index build
    paths["M"] = "";
}

int main(int argc, char **argv)
{
    std::cout << "Executing search_intersection.cpp" << std::endl;
    for (int i = 0; i < argc; i++)
    {
        std::string arg = argv[i];
        if (arg == "--data_path")
            paths["data_vector"] = argv[i + 1];
        if (arg == "--query_path")
            paths["query_vector"] = argv[i + 1];
        if (arg == "--range_saveprefix")
            paths["range_saveprefix"] = argv[i + 1];
        if (arg == "--groundtruth_saveprefix")
            paths["groundtruth_saveprefix"] = argv[i + 1];
        if (arg == "--index_file")
            paths["index"] = argv[i + 1];
        if (arg == "--result_saveprefix")
            paths["result_saveprefix"] = argv[i + 1];
        if (arg == "--baserange_saveprefix")
            paths["baserange_saveprefix"] = argv[i + 1];
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
    // storage.LoadQueryRange(paths["range_saveprefix"]);
    std::cout << "start load groundtruth " << '\n';
    storage.LoadGroundtruth(paths["groundtruth_saveprefix"]);
    std::cout << "end load groundtruth " << '\n';
    storage.LoadData(paths["data_vector"]);
    std::cout << "start init search " << '\n';
    std::cout << paths["data_vector"] << '\n';
    std::cout << paths["index"] << '\n';
     MultiSegmentTreeGraph::MSTGIndex<float> index(&storage, paths["index"]);
    std::cout << "end init search " << '\n';
    // searchefs can be adjusted
    std::vector<int> SearchEF;
    if (storage.dim == 128)
    {
        SearchEF = {10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 70, 80, 90, 100, 120, 140, 160, 180, 250, 350};
    }
    else if (storage.dim == 200)
    {
        SearchEF = {10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 70, 80, 90, 100, 120, 140, 160, 180, 250, 350};
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
            380, 480, 600, 750, 950, 1150, 1350, 1600};
    }

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

    std::string savepath = paths["result_saveprefix"] + get_current_timestamp() + ".csv";
    CheckPath(savepath);
    std::ofstream outfile(savepath);
    if (!outfile.is_open())
        throw Exception("cannot open " + savepath);

    index.IntersectionSearch(SearchEF, outfile, M, query_K, 3);
    outfile.close();
}