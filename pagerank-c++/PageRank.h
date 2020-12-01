//
// Created by 江玥 on 2020/4/27.
//

#ifndef PAGERANK_PAGERANK_H
#define PAGERANK_PAGERANK_H

#include <iostream>
#include <fstream>
#include <sys/stat.h>
#include <sys/types.h>
#include <set>
#include <vector>
#include <map>
#include <cmath>
#include <algorithm>
#include <dirent.h>
#include <numeric>
#include <pthread.h>

using namespace std;

class PageRank {
private:
    string data_file;
    string block_prefix;
    string res_file;
    int block_size;
    int block_num;
    map<int, int> src2index;
    map<int, int> index2src;
    long double beta;
    int max_iterations;

public:
    PageRank();
    PageRank(string _data_file, string _block_prefix, string _res_file, int _block_size, double beta, int _max_iterations);
    ~PageRank();
    void get_index();
    int get_node_num();
    void divide2block();
    vector<int> get_out_degree();
    vector<vector<int>> get_block_matrix(int num);
    void pagerank();
    void main();
};


#endif //PAGERANK_PAGERANK_H
