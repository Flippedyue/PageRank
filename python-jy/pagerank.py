from memory_profiler import profile
import numpy as np
import time
import os


SRC_FILE = 'WikiData.txt'
BLOCK_PREFIX = 'block/block_'
BLOCK_SIZE = 500
RESULT_FILE = 'result.txt'


class PageRank:
    def __init__(self, src_file, block_prefix, block_size, res_file):
        """初始化PageRank

        1.对一些变量进行赋值
        2.获得链接的映射index——src2index index2src
        3.对原始data分块

        :param src_file: 原始data文件
        :param block_prefix: 分块文件前缀
        :param block_size: 分块大小
        :param res_file: 结果文件
        """
        self.data_file = src_file
        self.src2index = dict()
        self.index2src = dict()
        self.block_path = block_prefix.split('/')[0]
        self.block_prefix = block_prefix
        self.block_size = block_size
        self.block_num = 0
        self.max_iterations = 1000000
        self.res_file = res_file
        self.total_nodes = 0
        self.total_edges = 0
        self.get_index()
        self.divide2block()

    # def get_index(self):
    #     """获得原始data <-> index之间的映射
    #     self.src2index
    #     self.index2src
    #     """
    #     i = 0
    #     with open(self.data_file, 'r', encoding='utf-8') as f:
    #         for line in f:
    #             x, y = line.split()
    #             x = int(x)
    #             y = int(y)
    #             if x not in self.src2index:
    #                 self.src2index[x] = i
    #                 self.index2src[i] = x
    #                 i += 1
    #             if y not in self.src2index:
    #                 self.src2index[y] = i
    #                 self.index2src[i] = y
    #                 i += 1
    #             self.total_edges += 1
    #     self.total_nodes = i

    def get_index(self):
        nodes_set = set()
        with open(self.data_file, 'r', encoding='utf-8') as f:
            for line in f:
                x, y = line.split()
                nodes_set.add(int(x))
                nodes_set.add(int(y))
                self.total_edges += 1
        index = 0
        for nodes in nodes_set:
            self.src2index[nodes] = index
            self.index2src[index] = nodes
            index += 1
        self.total_nodes = len(nodes_set)
        print('get index finished')

    def get_out_degree(self) -> np.array:
        """获取所有链接的出度

        :return: out_degree
        """
        out_degree = np.zeros(self.total_nodes)
        with open(self.data_file, 'r', encoding='utf-8') as f:
            for line in f:
                out_degree[int(self.src2index[int(line.split()[0])])] += 1
        return out_degree

    def divide2block(self):
        """对原始data进行分块操作"""
        if not os.path.exists(self.block_path):
            os.mkdir(self.block_path)
        src_file = open(self.data_file, 'r')
        self.block_num = int(self.total_nodes / self.block_size)
        if self.total_nodes % self.block_size != 0:
            self.block_num += 1

        block_files = []
        for i in range(self.block_num):
            block_file = open(self.block_prefix + str(i) + '.txt', 'w')
            block_files.append(block_file)

        for line in src_file:
            x, y = line.split()
            x = int(x)
            y = int(y)
            block_files[int(self.src2index[y] / self.block_size)].write(str(self.src2index[x]) + '  '
                                                                        + str(self.src2index[y]) + '\n')

        src_file.close()
        for i in range(self.block_num):
            block_files[i].close()
        print('divide to block finished')

    def get_block_matrix(self, num) -> np.array:
        """获得分块后的数据（出点 - 入点）

        :param num: 块编号

        :return: matrix —— [index[out_link] index[in_link]]
        """
        matrix = dict()
        with open(self.block_prefix + str(num) + '.txt', 'r') as f:
            for line in f:
                x, y = line.split()
                if int(x) in matrix:
                    matrix[int(x)].add(int(y))
                else:
                    matrix[int(x)] = {int(y)}
        return matrix

    @profile
    def block_page_rank(self, beta) -> np.array:
        """分块的pagerank算法

        :param beta: 占比

        :return: 最终处于稳定状态的特征向量
        """
        out_degree = self.get_out_degree()
        vec_old = np.ones(self.total_nodes) / self.total_nodes
        while True:
            vec_new = np.zeros(self.total_nodes)
            for i in range(self.block_num):
                vec = np.zeros(min(self.block_size, self.total_nodes - i * self.block_size))
                matrix = self.get_block_matrix(i)
                for _out in matrix:
                    for _in in matrix[_out]:
                        vec[_in % self.block_size] += vec_old[_out] * beta / out_degree[_out]
                vec_new[i * self.block_size: min((i+1) * self.block_size, self.total_nodes)] = vec

            vec_new += (1 - sum(vec_new)) / self.total_nodes
            diff = np.sqrt(sum((vec_new - vec_old) ** 2))

            vec_old = vec_new
            if diff <= 1e-5:
                break
        print('get vector finished')
        return vec_old

    def get_top(self, vector, num) -> list:
        """获取page score在前num个的link

        :param num: 前num个
        :param vector: 特征向量

        :return: top_val
        """
        sort_index = vector.argsort()[::-1][:num]
        vector.sort()
        top_vec = vector[::-1][:num]
        top_val = []
        for i in range(num):
            top_val.append([self.index2src[sort_index[i]], top_vec[i]])
        print(f"top {num} index - value : ")
        return top_val

    def write_back(self, top_val, res_file):
        """将结果写入结果文件并打印到终端中

        :param top_val: page score在前n个的[index, value]
        """
        with open(res_file, 'w') as f:
            for i in range(len(top_val)):
                f.write(str(top_val[i][0]) + '  ' + str(top_val[i][1]) + '\n')
                print(top_val[i][0], '-', top_val[i][1])

    def main(self):
        """pagerank主函数"""
        vector = self.block_page_rank(0.85)
        top_val = self.get_top(vector, 100)
        self.write_back(top_val, self.res_file)


def main():
    start = time.perf_counter()
    pr = PageRank(SRC_FILE, BLOCK_PREFIX, BLOCK_SIZE, RESULT_FILE)
    pr.main()
    end = time.perf_counter()
    print('time cost: ', str(end - start), 's')


if __name__ == '__main__':
    main()