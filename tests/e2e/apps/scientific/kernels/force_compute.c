#include "scicomp_types.h"

#include <math.h>

static void zero_forces(float *fx, float *fy, float *fz, int n) {
  for (int i = 0; i < n; ++i) {
    fx[i] = 0.0f;
    fy[i] = 0.0f;
    fz[i] = 0.0f;
  }
}

static void accumulate_pair(int i, int j, const float *px, const float *py,
                            const float *pz, const float *mass, float g,
                            float softening, float *fx, float *fy,
                            float *fz) {
  float dx = px[j] - px[i];
  float dy = py[j] - py[i];
  float dz = pz[j] - pz[i];
  float dist2 = dx * dx + dy * dy + dz * dz + softening * softening;
  float inv_dist = 1.0f / sqrtf(dist2);
  float inv_dist3 = inv_dist * inv_dist * inv_dist;
  float scale = g * mass[i] * mass[j] * inv_dist3;
  float fx_ij = dx * scale;
  float fy_ij = dy * scale;
  float fz_ij = dz * scale;
  fx[i] += fx_ij;
  fy[i] += fy_ij;
  fz[i] += fz_ij;
  fx[j] -= fx_ij;
  fy[j] -= fy_ij;
  fz[j] -= fz_ij;
}

void force_direct(const float *px, const float *py, const float *pz,
                  const float *mass, float *fx, float *fy, float *fz, int n,
                  float g, float softening) {
  if (!px || !py || !pz || !mass || !fx || !fy || !fz || n <= 0)
    return;

  zero_forces(fx, fy, fz, n);
  for (int i = 0; i < n; ++i) {
    for (int j = i + 1; j < n; ++j)
      accumulate_pair(i, j, px, py, pz, mass, g, softening, fx, fy, fz);
  }
}

void force_cutoff(const float *px, const float *py, const float *pz,
                  const float *mass, const int *offsets, const int *indices,
                  float *fx, float *fy, float *fz, int n, float g,
                  float softening) {
  if (!px || !py || !pz || !mass || !offsets || !indices || !fx || !fy ||
      !fz || n <= 0)
    return;

  zero_forces(fx, fy, fz, n);
  for (int i = 0; i < n; ++i) {
    for (int k = offsets[i]; k < offsets[i + 1]; ++k) {
      int j = indices[k];
      if (j <= i)
        continue;
      accumulate_pair(i, j, px, py, pz, mass, g, softening, fx, fy, fz);
    }
  }
}

void force_tree(const float *px, const float *py, const float *pz,
                const float *mass, const int *tree_offsets,
                const int *tree_indices, float *fx, float *fy, float *fz,
                int n, float g, float softening) {
  if (!px || !py || !pz || !mass || !fx || !fy || !fz || n <= 0)
    return;

  if (!tree_offsets || !tree_indices) {
    force_direct(px, py, pz, mass, fx, fy, fz, n, g, softening);
    return;
  }

  zero_forces(fx, fy, fz, n);
  for (int i = 0; i < n; ++i) {
    for (int k = tree_offsets[i]; k < tree_offsets[i + 1]; ++k) {
      int j = tree_indices[k];
      if (j <= i)
        continue;
      accumulate_pair(i, j, px, py, pz, mass, g, softening, fx, fy, fz);
    }
  }
}

void force_direct_unroll2(const float *px, const float *py, const float *pz,
                          const float *mass, float *fx, float *fy, float *fz,
                          int n, float g, float softening) {
  if (!px || !py || !pz || !mass || !fx || !fy || !fz || n <= 0)
    return;

  zero_forces(fx, fy, fz, n);
  for (int i = 0; i < n; ++i) {
    int j = i + 1;
    for (; j + 1 < n; j += 2) {
      accumulate_pair(i, j, px, py, pz, mass, g, softening, fx, fy, fz);
      accumulate_pair(i, j + 1, px, py, pz, mass, g, softening, fx, fy, fz);
    }
    for (; j < n; ++j)
      accumulate_pair(i, j, px, py, pz, mass, g, softening, fx, fy, fz);
  }
}
