#ifndef SCI_COMP_TYPES_H
#define SCI_COMP_TYPES_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  int nrows;
  int ncols;
  int nnz;
  const int *row_ptr;
  const int *col_idx;
  const float *values;
} ScicompCSRMatrix;

typedef struct {
  int nrows;
  int ncols;
  int max_nnz;
  const int *col_idx;
  const float *values;
} ScicompELLMatrix;

typedef struct {
  int rows;
  int cols;
  int halo_w;
} ScicompGrid2D;

typedef struct {
  int n;
  float *x;
  float *y;
  float *z;
  float *vx;
  float *vy;
  float *vz;
  float *fx;
  float *fy;
  float *fz;
  float *mass;
} ScicompParticleState;

typedef struct {
  int n;
  const int *offsets;
  const int *indices;
} ScicompNeighborList;

static inline int scicomp_grid_total_rows(ScicompGrid2D grid) {
  return grid.rows + 2 * grid.halo_w;
}

static inline int scicomp_grid_total_cols(ScicompGrid2D grid) {
  return grid.cols + 2 * grid.halo_w;
}

static inline size_t scicomp_grid_index(ScicompGrid2D grid, int row, int col) {
  return (size_t)row * (size_t)scicomp_grid_total_cols(grid) + (size_t)col;
}

#ifdef __cplusplus
}
#endif

#endif
