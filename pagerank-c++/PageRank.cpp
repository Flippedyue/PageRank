//
// Created by 江玥 on 2020/4/27.
//

#include "PageRank.h"


template <typename T>
vector<size_t> argsort(const vector<T> &v)
{
    // 建立下标数组
    vector<size_t> idx(v.size());
    iota(idx.begin(), idx.end(), 0);
    // 调用sort函数，匿名函数自动捕获待排序数组
    sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2) {return v[i1] > v[i2];});
    return idx;
}

PageRank::PageRank() {

}

PageRank::PageRank(string _data_file, string _block_prefix, string _res_file, int _block_size, double _beta,
                   int _max_iterations) : data_file(std::move(_data_file)), block_prefix(std::move(_block_prefix)),
                   res_file(std::move(_res_file)),block_size(_block_size), beta(_beta), max_iterations(_max_iterations) {
    block_num = 0;
    src2index.clear ();
    index2src.clear ();
}

PageRank::~PageRank() {

}

void PageRank::get_index() {
    ifstream in_file;
    in_file.open(data_file.c_str());
    int x, y, i = 0;
    while(in_file >> x >> y) {
        if (!src2index.count(x)) {
            src2index[x] = i;
            index2src[i] = x;
            i++;
        }
        if (!src2index.count(y)) {
            src2index[y] = i;
            index2src[i] = y;
            i++;
        }
    }
    in_file.close();
    ofstream out_file;
    out_file.open("index.txt");
    for(auto item: src2index) {
        out_file << item.first << "  " << item.second << endl;
    }
    out_file.close();
    cout << "get index finished" << endl;
}

int PageRank::get_node_num() {
    return src2index.size();
}

void PageRank::divide2block() {
    string block_path = "block";
    DIR *dp;
    if (!(dp = opendir(block_path.c_str()))) {
        mkdir(block_path.c_str(), S_IRWXU);
    }
    ifstream in_file;
    ofstream out_file;
    in_file.open(data_file);
    out_file.open((block_prefix + to_string (block_num++) + ".txt").c_str());
    int count = 0;
    string line;
    while (getline(in_file, line)) {
        if (count % block_size == 0 && count != 0) {
            out_file.close();
            out_file.open(block_prefix + to_string(block_num++) + ".txt");
        }
        out_file << line;
        count++;
    }
    in_file.close();
    out_file.close();
    cout << "divide to block finished" << endl;
}

vector<int> PageRank::get_out_degree() {
    vector<int> out_degree(get_node_num (), 0);
    ifstream in_file;
    in_file.open(data_file);
    ofstream out_file;
    out_file.open("out_degree.txt");
    int x, y;
    while (in_file >> x >> y) {
        out_degree[src2index[x]]++;
    }
    in_file.close();
    for (int i = 0; i < get_node_num (); ++i) {
        out_file << i << "  " << out_degree[i] << endl;
    }
    out_file.close();
    return out_degree;
}

vector<vector<int>> PageRank::get_block_matrix(int num) {
    vector<vector<int>> matrix;
    ifstream block_file;
    block_file.open(block_prefix + to_string (num) + ".txt");
    int x, y;
    while (block_file >> x >> y) {
        matrix.push_back({src2index[x], src2index[y]});
    }
    block_file.close();
    return matrix;
}

void PageRank::pagerank() {
    vector<int> out_degree = get_out_degree();
    vector<long double> vec(get_node_num (), 1.0 / get_node_num ());
    for (int i = 0; i < 100; ++i) {
        vector<long double> vec_new(get_node_num (), 0);

        #pragma omp parallel  for
        for (int j = 0; j < block_num; ++j) {
            vector<vector<int>> matrix = get_block_matrix (j);
            // vector<vector<int>> matrix = get_block_matrix (data);
            for (auto edge: matrix) {
                vec_new[edge[1]] += vec[edge[0]] * beta / out_degree[edge[0]];
            }
        }

        long double sum = 0;
        #pragma omp parallel for reduction(+:sum)
        for (int j = 0; j < vec_new.size(); j++)
            sum += vec_new[j];

        // vector<long double> vec_sub(get_node_num (), (1.0 - sum) / get_node_num());
        #pragma omp parallel for
        for (int j = 0; j < vec_new.size(); ++j) {
            vec_new[j] += (1.0 - sum) / get_node_num();
        }

        sum = 0;
        #pragma omp parallel for reduction(+:sum)
        for (int j = 0; j < vec_new.size(); ++j) {
            sum += pow(vec[j] - vec_new[j], 2);
        }
        vec = vec_new;
        sum = sqrt (sum);
        if (sum <= 1e-7) {
            break;
        }
    }
    vector<size_t> index = argsort(vec);
    sort(vec.begin(), vec.end(), greater<long double>());
    ofstream result;
    result.open(res_file.c_str());
    for (int i = 0; i < 100; ++i) {
        cout << index2src[index[i]] << " : " << vec[i] << endl;
        result << index2src[index[i]] << " : " << vec[i] << endl;
    }
    result.close();
}

void PageRank::main() {
    get_index ();
    divide2block ();
    pagerank ();
}

