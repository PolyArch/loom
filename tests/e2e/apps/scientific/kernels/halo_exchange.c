#include "scicomp_types.h"

static void copy_row(float *tile, int dst_row, int src_row, int total_cols) {
  for (int col = 0; col < total_cols; ++col)
    tile[(size_t)dst_row * (size_t)total_cols + (size_t)col] =
        tile[(size_t)src_row * (size_t)total_cols + (size_t)col];
}

void halo_row_only(float *tile, int rows, int cols, int halo_w) {
  if (!tile || rows <= 0 || cols <= 0 || halo_w <= 0)
    return;

  int total_rows = rows + 2 * halo_w;
  int total_cols = cols + 2 * halo_w;

  for (int h = 0; h < halo_w; ++h) {
    copy_row(tile, h, halo_w, total_cols);
    copy_row(tile, total_rows - 1 - h, total_rows - 1 - halo_w, total_cols);
  }
}

void halo_row_col(float *tile, int rows, int cols, int halo_w) {
  if (!tile || rows <= 0 || cols <= 0 || halo_w <= 0)
    return;

  int total_rows = rows + 2 * halo_w;
  int total_cols = cols + 2 * halo_w;

  halo_row_only(tile, rows, cols, halo_w);

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
