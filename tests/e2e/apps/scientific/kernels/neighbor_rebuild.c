#include "scicomp_types.h"

#include <math.h>

static int add_neighbor(int *indices, int max_neighbors, int cursor, int j) {
  if (cursor >= max_neighbors)
    return cursor;
  indices[cursor] = j;
  return cursor + 1;
}

int rebuild_cell_list(const float *px, const float *py, const float *pz, int n,
                      float cutoff, int *offsets, int *indices,
                      int max_neighbors) {
  if (!px || !py || !pz || !offsets || !indices || n <= 0 || max_neighbors <= 0)
    return 0;

  float cutoff2 = cutoff * cutoff;
  int cursor = 0;
  for (int i = 0; i < n; ++i) {
    offsets[i] = cursor;
    for (int j = 0; j < n; ++j) {
      if (i == j)
        continue;
      float dx = px[j] - px[i];
      float dy = py[j] - py[i];
      float dz = pz[j] - pz[i];
      float dist2 = dx * dx + dy * dy + dz * dz;
      if (dist2 <= cutoff2)
        cursor = add_neighbor(indices, max_neighbors, cursor, j);
    }
  }
  offsets[n] = cursor;
  return cursor;
}

int rebuild_verlet_list(const float *px, const float *py, const float *pz, int n,
                        float cutoff, float skin, int *offsets, int *indices,
                        int max_neighbors) {
  return rebuild_cell_list(px, py, pz, n, cutoff + skin, offsets, indices,
                           max_neighbors);
}
