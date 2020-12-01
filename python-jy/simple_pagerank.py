from memory_profiler import profile
import numpy as np
import time
import sys

src2index = dict()
index2src = dict()


def get_index(data_file):
    """获得原始data <-> index之间的映射

    src2index 原始编号 =》现有编号
    index2src 现有编号 =》原始编号
    """
    nodes = set()
    with open(data_file, 'r', encoding='utf-8') as f:
        for line in f:
            x, y = line.split()
            nodes.add(int(x))
            nodes.add(int(y))
    i = 0
    for node in nodes:
        src2index[node] = i
        index2src[i] = node
        i += 1


def get_nodes_num():
    return len(src2index)


def get_out_degree(data_file):
    """

    :param data_file:
    :return:
    """
    out_degree = np.zeros(get_nodes_num())
    with open(data_file, 'r', encoding='utf-8') as f:
        for line in f:
            out_degree[int(src2index[int(line.split()[0])])] += 1
    return out_degree


def load_data(path):
    get_index(path)
    out_degree = get_out_degree(path)
    data = np.zeros(shape=(get_nodes_num(), get_nodes_num()))
    f = open(path, 'r')
    for line in f:
        x, y = line.split()
        data[src2index[int(y)]][src2index[int(x)]] = 1 / out_degree[src2index[int(x)]]
    f.close()
    for i in range(len(out_degree)):
        if out_degree[i] == 0:
            data[:, i] = 1.0 / len(out_degree)
    return data


def pagerank(M, num_iterations: int = 100, d: float = 0.85):
    """PageRank核心算法

    Parameters
    ----------
    M : numpy array
        adjacency matrix where M_i,j represents the link from 'j' to 'i', such that for all 'j'
        sum(i, M_i,j) = 1
    num_iterations : int, optional
        number of iterations, by default 100
    d : float, optional
        damping factor, by default 0.85

    Returns
    -------
    numpy array
        a vector of ranks such that v_i is the i-th rank from [0, 1],
        v sums to 1

    """
    # print(N)
    v = np.ones(get_nodes_num()) / get_nodes_num()

    for i in range(num_iterations):
        v_new = M @ v * d + (1 - d) / get_nodes_num()
        s = np.sqrt(sum((v - v_new) ** 2))
        v = v_new
        if s < 1e-5:
            break
    return v


# @profile
def get_top():
    """获取前100个PageRank值
    """
    M = load_data("WikiData.txt")
    print(sys.getsizeof(M))
    v = pagerank(M, 100, 0.85)
    sort_index = v.argsort()[::-1][:100]
    v.sort()
    top_vec = v[::-1][:100]
    top_val = []
    for i in range(100):
        top_val.append([index2src[sort_index[i]], top_vec[i]])
    res = open('res.txt', 'w')
    for i in range(100):
        print(top_val[i][0], top_val[i][1])
        res.write(str(top_val[i][0]) + '   ' + str(top_val[i][1]) + '\n')
    res.close()


# @profile()
def main():
    start = time.perf_counter()
    get_top()
    end = time.perf_counter()
    print('time cost: ', str(end - start), 's')


if __name__ == "__main__":
    main()