#include "scicomp_types.h"

static int scicomp_interior_rows(int rows, int halo_w) {
  return rows + 2 * halo_w;
}

static int scicomp_interior_cols(int cols, int halo_w) {
  return cols + 2 * halo_w;
}

void stencil_5pt(const float *in, float *out, int rows, int cols,
                 int halo_w, float factor) {
  if (!in || !out || rows <= 0 || cols <= 0 || halo_w <= 0)
    return;

  int total_rows = scicomp_interior_rows(rows, halo_w);
  int total_cols = scicomp_interior_cols(cols, halo_w);

  for (int i = 0; i < total_rows * total_cols; ++i)
    out[i] = in[i];

  for (int row = halo_w; row < rows + halo_w; ++row) {
    for (int col = halo_w; col < cols + halo_w; ++col) {
      float center = in[(size_t)row * (size_t)total_cols + (size_t)col];
      float sum = in[(size_t)(row - 1) * (size_t)total_cols + (size_t)col] +
                  in[(size_t)(row + 1) * (size_t)total_cols + (size_t)col] +
                  in[(size_t)row * (size_t)total_cols + (size_t)(col - 1)] +
                  in[(size_t)row * (size_t)total_cols + (size_t)(col + 1)] -
                  4.0f * center;
      out[(size_t)row * (size_t)total_cols + (size_t)col] = sum * factor;
    }
  }
}

void stencil_9pt(const float *in, float *out, int rows, int cols, int halo_w,
                 float factor) {
  if (!in || !out || rows <= 0 || cols <= 0 || halo_w <= 0)
    return;

  int total_rows = scicomp_interior_rows(rows, halo_w);
  int total_cols = scicomp_interior_cols(cols, halo_w);

  for (int i = 0; i < total_rows * total_cols; ++i)
    out[i] = in[i];

  for (int row = halo_w; row < rows + halo_w; ++row) {
    for (int col = halo_w; col < cols + halo_w; ++col) {
      float center = in[(size_t)row * (size_t)total_cols + (size_t)col];
      float orth = in[(size_t)(row - 1) * (size_t)total_cols + (size_t)col] +
                   in[(size_t)(row + 1) * (size_t)total_cols + (size_t)col] +
                   in[(size_t)row * (size_t)total_cols + (size_t)(col - 1)] +
                   in[(size_t)row * (size_t)total_cols + (size_t)(col + 1)];
      float diag = in[(size_t)(row - 1) * (size_t)total_cols +
                      (size_t)(col - 1)] +
                   in[(size_t)(row - 1) * (size_t)total_cols +
                      (size_t)(col + 1)] +
                   in[(size_t)(row + 1) * (size_t)total_cols +
                      (size_t)(col - 1)] +
                   in[(size_t)(row + 1) * (size_t)total_cols +
                      (size_t)(col + 1)];
      out[(size_t)row * (size_t)total_cols + (size_t)col] =
          (orth + 0.5f * diag - 6.0f * center) * factor;
    }
  }
}

void stencil_5pt_unroll2(const float *in, float *out, int rows, int cols,
                         int halo_w, float factor) {
  if (!in || !out || rows <= 0 || cols <= 0 || halo_w <= 0)
    return;

  int total_rows = scicomp_interior_rows(rows, halo_w);
  int total_cols = scicomp_interior_cols(cols, halo_w);

  for (int i = 0; i < total_rows * total_cols; ++i)
    out[i] = in[i];

  for (int row = halo_w; row < rows + halo_w; ++row) {
    int col = halo_w;
    for (; col + 1 < cols + halo_w; col += 2) {
      for (int lane = 0; lane < 2; ++lane) {
        int c = col + lane;
        float center = in[(size_t)row * (size_t)total_cols + (size_t)c];
        float sum = in[(size_t)(row - 1) * (size_t)total_cols + (size_t)c] +
                    in[(size_t)(row + 1) * (size_t)total_cols + (size_t)c] +
                    in[(size_t)row * (size_t)total_cols + (size_t)(c - 1)] +
                    in[(size_t)row * (size_t)total_cols + (size_t)(c + 1)] -
                    4.0f * center;
        out[(size_t)row * (size_t)total_cols + (size_t)c] = sum * factor;
      }
    }
    for (; col < cols + halo_w; ++col) {
      float center = in[(size_t)row * (size_t)total_cols + (size_t)col];
      float sum = in[(size_t)(row - 1) * (size_t)total_cols + (size_t)col] +
                  in[(size_t)(row + 1) * (size_t)total_cols + (size_t)col] +
                  in[(size_t)row * (size_t)total_cols + (size_t)(col - 1)] +
                  in[(size_t)row * (size_t)total_cols + (size_t)(col + 1)] -
                  4.0f * center;
      out[(size_t)row * (size_t)total_cols + (size_t)col] = sum * factor;
    }
  }
}
