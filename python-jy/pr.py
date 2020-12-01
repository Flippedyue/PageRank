import numpy as np
import time
import os

SRC_FILE = 'WikiData.txt'
BLOCK_PREFIX = 'block/block_'
BLOCK_SIZE = 2000
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

    def get_index(self):
        """获得原始data <-> index之间的映射
        self.src2index
        self.index2src
        """
        i = 0
        with open(self.data_file, 'r', encoding='utf-8') as f:
            for line in f:
                x, y = line.split()
                x = int(x)
                y = int(y)
                if x not in self.src2index:
                    self.src2index[x] = i
                    self.index2src[i] = x
                    i += 1
                if y not in self.src2index:
                    self.src2index[y] = i
                    self.index2src[i] = y
                    i += 1
                self.total_edges += 1
        self.total_nodes = i

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

        count = 0
        src_file = open(self.data_file, 'r')
        block_file = open(self.block_prefix + str(self.block_num) + '.txt', 'w')
        self.block_num += 1
        for line in src_file:
            if count % self.block_size == 0 and count != 0:
                block_file.close()
                block_file = open(self.block_prefix + str(self.block_num) + '.txt', 'w')
                self.block_num += 1
            block_file.write(line)
            count += 1
        src_file.close()
        block_file.close()

    def get_block_matrix(self, num: int) -> np.array:
        """获得分块后的数据（入点 - 出点）

        :param num: 块编号

        :return: data —— [in_link out_link]
        """
        matrix = []
        with open(self.block_prefix + str(num) + '.txt', 'r') as f:
            for line in f:
                x, y = line.split()
                matrix.append([self.src2index[int(x)], self.src2index[int(y)]])
        matrix = np.array(matrix)
        return matrix

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
                matrix = self.get_block_matrix(i)
                for edge in matrix:
                    vec_new[edge[1]] += vec_old[edge[0]] * beta / out_degree[edge[0]]
            vec_new += (1 - sum(vec_new)) / self.total_nodes
            diff = np.sqrt(sum((vec_new - vec_old) ** 2))
            vec_old = vec_new
            if diff <= 1e-5:
                break
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
        for i in range(num):
            print(top_val[i][0], top_val[i][1])
        return top_val

    def write_result(self, top_val, res_file):
        """将结果写入结果文件中

        :param top_val: page score在前n个的[index, value]
        """
        with open(res_file, 'w') as f:
            for i in range(len(top_val)):
                f.write(str(top_val[i][0]) + '  ' + str(top_val[i][1]) + '\n')

    def main(self):
        """pagerank主函数"""
        vector = self.block_page_rank(0.85)
        top_val = self.get_top(vector, 100)
        self.write_result(top_val, self.res_file)


if __name__ == '__main__':
    start = time.perf_counter()
    pr = PageRank(SRC_FILE, BLOCK_PREFIX, BLOCK_SIZE, RESULT_FILE)
    pr.main()
    end = time.perf_counter()
    print('time cost: ', str(end - start), 's')
