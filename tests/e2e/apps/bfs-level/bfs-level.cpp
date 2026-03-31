#include <cstdio>

int bfs_level(const int *rowPtr, const int *colIdx, const int *frontier,
              int frontierSize, int *visited, int *nextFrontier) {
  int nextSize = 0;
  for (int idx = 0; idx < frontierSize; ++idx) {
    int node = frontier[idx];
    for (int edge = rowPtr[node]; edge < rowPtr[node + 1]; ++edge) {
      int dst = colIdx[edge];
      if (!visited[dst]) {
        visited[dst] = 1;
        nextFrontier[nextSize++] = dst;
      }
    }
  }
  return nextSize;
}

int main() {
  int rowPtr[6] = {0, 2, 4, 5, 6, 6};
  int colIdx[6] = {1, 2, 3, 4, 4, 0};
  int frontier[2] = {0, 1};
  int visited[5] = {1, 1, 0, 0, 0};
  int nextFrontier[5] = {};
  int golden[3] = {2, 3, 4};

  int nextSize = bfs_level(rowPtr, colIdx, frontier, 2, visited, nextFrontier);
  if (nextSize != 3) {
    std::printf("FAIL bfs-size %d 3\n", nextSize);
    return 1;
  }
  for (int idx = 0; idx < nextSize; ++idx) {
    if (nextFrontier[idx] != golden[idx]) {
      std::printf("FAIL bfs-node %d %d %d\n", idx, nextFrontier[idx], golden[idx]);
      return 1;
    }
  }
  std::puts("PASS");
  return 0;
}
