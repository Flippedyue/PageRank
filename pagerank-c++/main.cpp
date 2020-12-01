#include "PageRank.h"

using namespace std;

int main() {
    PageRank pr("WikiData.txt", "block/block_", "res.txt", 20000, 0.85, 100);
    clock_t start = clock();
    pr.main();
    clock_t end = clock();
    cout << "use "
         << (end - start) / 1000000 << " s" << endl;
    return 0;
}
