#include "scicomp_types.h"

#include <math.h>
#include <stdio.h>

void axpy_basic(float alpha, const float *x, float *y, int n);
void spmv_csr(const int *row_ptr, const int *col_idx, const float *values,
              const float *x, float *y, int nrows);
void stencil_5pt(const float *in, float *out, int rows, int cols, int halo_w,
                 float factor);
void force_direct(const float *px, const float *py, const float *pz,
                  const float *mass, float *fx, float *fy, float *fz, int n,
                  float g, float softening);
float energy_ke_only(const float *vx, const float *vy, const float *vz,
                     const float *mass, int n);

static int nearly_equal(float lhs, float rhs, float eps) {
  float diff = lhs - rhs;
  if (diff < 0.0f)
    diff = -diff;
  return diff <= eps;
}

static int check_axpy(void) {
  float x[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  float y[4] = {10.0f, 20.0f, 30.0f, 40.0f};
  axpy_basic(2.0f, x, y, 4);
  return nearly_equal(y[0], 12.0f, 1e-6f) && nearly_equal(y[3], 48.0f, 1e-6f);
}

static int check_spmv(void) {
  const int row_ptr[3] = {0, 2, 4};
  const int col_idx[4] = {0, 1, 0, 1};
  const float values[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  const float x[2] = {1.0f, 1.0f};
  float y[2] = {0.0f, 0.0f};
  spmv_csr(row_ptr, col_idx, values, x, y, 2);
  return nearly_equal(y[0], 3.0f, 1e-6f) && nearly_equal(y[1], 7.0f, 1e-6f);
}

static int check_stencil(void) {
  float in[16] = {0.0f};
  float out[16] = {0.0f};
  in[5] = 1.0f;
  in[6] = 2.0f;
  in[9] = 3.0f;
  in[10] = 4.0f;
  stencil_5pt(in, out, 2, 2, 1, 1.0f);
  return nearly_equal(out[5], 1.0f, 1e-6f);
}

static int check_nbody(void) {
  const float px[2] = {0.0f, 1.0f};
  const float py[2] = {0.0f, 0.0f};
  const float pz[2] = {0.0f, 0.0f};
  const float mass[2] = {1.0f, 1.0f};
  float fx[2] = {0.0f, 0.0f};
  float fy[2] = {0.0f, 0.0f};
  float fz[2] = {0.0f, 0.0f};
  force_direct(px, py, pz, mass, fx, fy, fz, 2, 1.0f, 0.0f);
  return nearly_equal(fx[0], 1.0f, 1e-6f) &&
         nearly_equal(fx[1], -1.0f, 1e-6f) &&
         nearly_equal(fy[0], 0.0f, 1e-6f) && nearly_equal(fy[1], 0.0f, 1e-6f) &&
         nearly_equal(fz[0], 0.0f, 1e-6f) && nearly_equal(fz[1], 0.0f, 1e-6f);
}

int main(void) {
  int ok = check_axpy() && check_spmv() && check_stencil();

  {
    const float vx[2] = {1.0f, 0.0f};
    const float vy[2] = {0.0f, 1.0f};
    const float vz[2] = {0.0f, 0.0f};
    const float mass[2] = {2.0f, 3.0f};
    float ke = energy_ke_only(vx, vy, vz, mass, 2);
    ok = ok && nearly_equal(ke, 2.5f, 1e-6f);
  }

  ok = ok && check_nbody();

  printf("scientific kernels smoke: %s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
