"""PageRank algorithm with explicit number of iterations.

Returns
-------
ranking of nodes (pages) in the adjacency matrix

"""
from scipy.sparse import coo_matrix
from memory_profiler import profile
import numpy as np
import time
import sys

src2index = dict()
index2src = dict()
row = []
col = []
data = []


def get_index(data_file):
    """获得原始data <-> index之间的映射
    self.src2index
    self.index2src
    """
    i = 0
    edges = dict()
    with open(data_file, 'r', encoding='utf-8') as f:
        for line in f:
            x, y = line.split()
            x = int(x)
            y = int(y)
            if x not in src2index:
                src2index[x] = i
                index2src[i] = x
                i += 1
            if y not in src2index:
                src2index[y] = i
                index2src[i] = y
                i += 1
            if src2index[x] not in edges:
                edges[src2index[x]] = set()
            edges[src2index[x]].add(src2index[y])
    # print(sys.getsizeof(edges))
    return edges


def pre_process(edges):
    for i in range(len(src2index)):
        if i in edges:
            for j in edges[i]:
                row.append(j)
                col.append(i)
                data.append(1 / len(edges[i]))
        else:
            for j in range(len(src2index)):
                row.append(j)
                col.append(i)
                data.append(1 / len(src2index))


def load_data(path):
    edges = get_index(path)
    pre_process(edges)
    coo_M = coo_matrix((data, (row, col)), shape=(len(src2index), len(src2index)))
    return coo_M


def pagerank(M, num_iterations: int = 1000000, d: float = 0.85):
    """PageRank: The trillion dollar algorithm.

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
    N = M.shape[1]
    v = np.ones(N) / N
    for i in range(num_iterations):
        v_new = M @ v * d + (1 - d) / N
        s = 0
        for i in range(0, len(v)):
            s += abs(v[i] - v_new[i])
        v = v_new
        if s < 1e-5:
            break
    return v


@profile()
def main():
    start = time.perf_counter()
    M = load_data("WikiData.txt")
    # print(sys.getsizeof(M))
    v = pagerank(M)
    sort_index = v.argsort()[::-1][:100]
    v.sort()
    top_vec = v[::-1][:100]
    top_val = []
    for i in range(100):
        top_val.append([index2src[sort_index[i]], top_vec[i]])
    for i in range(100):
        print(top_val[i][0], top_val[i][1])
    end = time.perf_counter()
    print('time cost: ', str(end - start), 's')


if __name__ == "__main__":
    main()