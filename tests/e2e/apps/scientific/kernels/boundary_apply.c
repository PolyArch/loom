#include "scicomp_types.h"

void boundary_dirichlet(float *tile, int rows, int cols, int halo_w,
                        float value) {
  if (!tile || rows <= 0 || cols <= 0 || halo_w <= 0)
    return;

  int total_rows = rows + 2 * halo_w;
  int total_cols = cols + 2 * halo_w;

  for (int row = 0; row < total_rows; ++row) {
    for (int h = 0; h < halo_w; ++h) {
      tile[(size_t)row * (size_t)total_cols + (size_t)h] = value;
      tile[(size_t)row * (size_t)total_cols +
           (size_t)(total_cols - 1 - h)] = value;
    }
  }
  for (int col = 0; col < total_cols; ++col) {
    for (int h = 0; h < halo_w; ++h) {
      tile[(size_t)h * (size_t)total_cols + (size_t)col] = value;
      tile[(size_t)(total_rows - 1 - h) * (size_t)total_cols +
           (size_t)col] = value;
    }
  }
}

void boundary_neumann(float *tile, int rows, int cols, int halo_w) {
  if (!tile || rows <= 0 || cols <= 0 || halo_w <= 0)
    return;

  int total_rows = rows + 2 * halo_w;
  int total_cols = cols + 2 * halo_w;

  for (int h = 0; h < halo_w; ++h) {
    int top_dst = h;
    int top_src = halo_w;
    int bot_dst = total_rows - 1 - h;
    int bot_src = total_rows - 1 - halo_w;
    for (int col = 0; col < total_cols; ++col) {
      tile[(size_t)top_dst * (size_t)total_cols + (size_t)col] =
          tile[(size_t)top_src * (size_t)total_cols + (size_t)col];
      tile[(size_t)bot_dst * (size_t)total_cols + (size_t)col] =
          tile[(size_t)bot_src * (size_t)total_cols + (size_t)col];
    }
  }

  for (int row = 0; row < total_rows; ++row) {
    for (int h = 0; h < halo_w; ++h) {
      tile[(size_t)row * (size_t)total_cols + (size_t)h] =
          tile[(size_t)row * (size_t)total_cols + (size_t)halo_w];
      tile[(size_t)row * (size_t)total_cols +
           (size_t)(total_cols - 1 - h)] =
          tile[(size_t)row * (size_t)total_cols +
               (size_t)(total_cols - 1 - halo_w)];
    }
  }
}
