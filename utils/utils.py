import numpy as np
from joblib import Parallel, delayed
import struct

def calc_recall_for_single_query(res_i, gt_i, X, Y):
    tp = sum(res_i[i] in gt_i[:Y] for i in range(X))
    return tp / X


def calc_recall(res, gt, x, y):
    n_res, k_res = res.shape
    n_gt, k_gt = gt.shape
    assert n_res == n_gt, "n_res and n_gt should be equal"
    assert k_gt >= y, "k_res must be at least Y"

    recall_list = Parallel(n_jobs=-1)(
        delayed(calc_recall_for_single_query)(res[i], gt[i], x, y) for i in range(n_res)
    )
    recall = np.mean(recall_list)
    # print(f"recall{x}@{y}: {recall:.4f}")
    return recall
def ibin_read(fname):
    total_file = np.fromfile(fname, dtype="uint32")
    num = total_file[0]
    dim = total_file[1]
    return total_file[2:].reshape(num, dim).copy()

def fbin_read(fname):
    return ibin_read(fname).view("float32")

def query_range_read(fname):
    total_file = np.fromfile(fname, dtype="int32");
    total_file = total_file.reshape(int(total_file.shape[0]/2),2);
    print(f"Detected range_nb = {int(total_file.shape[0])}, topk = {2}")
    return total_file


def load_groundtruth_numpy_fast(filename):
    # 读取整个二进制文件
    with open(filename, 'rb') as f:
        buffer = f.read()

    # 提取第一个 topk
    first_topk = struct.unpack_from('i', buffer, 0)[0]
    topk = first_topk
    entry_size = (1 + topk) * 4  # 每条记录占用字节数（1个 size + topk 个 int）

    # 计算 query 数量
    file_size = len(buffer)
    if file_size % entry_size != 0:
        raise ValueError("File size is not aligned with fixed topk format.")

    query_nb = file_size // entry_size
    print(f"Detected query_nb = {query_nb}, topk = {topk}")

    # 初始化 numpy array
    gt_array = np.empty((query_nb, topk), dtype=np.int32)

    # 逐条提取 groundtruth ID 列表
    offset = 0
    for i in range(query_nb):
        size = struct.unpack_from('i', buffer, offset)[0]
        if size != topk:
            raise ValueError(f"Inconsistent topk size at query {i}: expected {topk}, got {size}")
        offset += 4
        gt_array[i] = np.frombuffer(buffer, dtype=np.int32, count=topk, offset=offset)
        offset += topk * 4

    return gt_array