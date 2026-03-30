#include "scicomp_types.h"

#include <math.h>

float energy_ke_only(const float *vx, const float *vy, const float *vz,
                     const float *mass, int n) {
  if (!vx || !vy || !vz || !mass || n <= 0)
    return 0.0f;

  float total = 0.0f;
  for (int i = 0; i < n; ++i) {
    float speed2 = vx[i] * vx[i] + vy[i] * vy[i] + vz[i] * vz[i];
    total += 0.5f * mass[i] * speed2;
  }
  return total;
}

float energy_ke_pe(const float *px, const float *py, const float *pz,
                   const float *vx, const float *vy, const float *vz,
                   const float *mass, const int *offsets, const int *indices,
                   int n, float g, float softening) {
  if (!px || !py || !pz || !vx || !vy || !vz || !mass || n <= 0)
    return 0.0f;

  float total = energy_ke_only(vx, vy, vz, mass, n);
  float soft2 = softening * softening;

  if (offsets && indices) {
    for (int i = 0; i < n; ++i) {
      for (int k = offsets[i]; k < offsets[i + 1]; ++k) {
        int j = indices[k];
        if (j <= i)
          continue;
        float dx = px[j] - px[i];
        float dy = py[j] - py[i];
        float dz = pz[j] - pz[i];
        float dist = sqrtf(dx * dx + dy * dy + dz * dz + soft2);
        if (dist > 0.0f)
          total -= g * mass[i] * mass[j] / dist;
      }
    }
    return total;
  }

  for (int i = 0; i < n; ++i) {
    for (int j = i + 1; j < n; ++j) {
      float dx = px[j] - px[i];
      float dy = py[j] - py[i];
      float dz = pz[j] - pz[i];
      float dist = sqrtf(dx * dx + dy * dy + dz * dz + soft2);
      if (dist > 0.0f)
        total -= g * mass[i] * mass[j] / dist;
    }
  }
  return total;
}
