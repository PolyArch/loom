#include "scicomp_types.h"

void update_verlet(float *px, float *py, float *pz, float *vx, float *vy,
                   float *vz, const float *fx, const float *fy,
                   const float *fz, const float *mass, int n, float dt) {
  if (!px || !py || !pz || !vx || !vy || !vz || !fx || !fy || !fz || !mass ||
      n <= 0)
    return;

  float half_dt = 0.5f * dt;
  for (int i = 0; i < n; ++i) {
    float inv_mass = mass[i] != 0.0f ? 1.0f / mass[i] : 1.0f;
    float ax = fx[i] * inv_mass;
    float ay = fy[i] * inv_mass;
    float az = fz[i] * inv_mass;
    vx[i] += half_dt * ax;
    vy[i] += half_dt * ay;
    vz[i] += half_dt * az;
    px[i] += dt * vx[i];
    py[i] += dt * vy[i];
    pz[i] += dt * vz[i];
    vx[i] += half_dt * ax;
    vy[i] += half_dt * ay;
    vz[i] += half_dt * az;
  }
}

void update_verlet_unroll2(float *px, float *py, float *pz, float *vx,
                           float *vy, float *vz, const float *fx,
                           const float *fy, const float *fz,
                           const float *mass, int n, float dt) {
  if (!px || !py || !pz || !vx || !vy || !vz || !fx || !fy || !fz || !mass ||
      n <= 0)
    return;

  float half_dt = 0.5f * dt;
  int i = 0;
  for (; i + 1 < n; i += 2) {
    for (int lane = 0; lane < 2; ++lane) {
      int idx = i + lane;
      float inv_mass = mass[idx] != 0.0f ? 1.0f / mass[idx] : 1.0f;
      float ax = fx[idx] * inv_mass;
      float ay = fy[idx] * inv_mass;
      float az = fz[idx] * inv_mass;
      vx[idx] += half_dt * ax;
      vy[idx] += half_dt * ay;
      vz[idx] += half_dt * az;
      px[idx] += dt * vx[idx];
      py[idx] += dt * vy[idx];
      pz[idx] += dt * vz[idx];
      vx[idx] += half_dt * ax;
      vy[idx] += half_dt * ay;
      vz[idx] += half_dt * az;
    }
  }
  for (; i < n; ++i) {
    float inv_mass = mass[i] != 0.0f ? 1.0f / mass[i] : 1.0f;
    float ax = fx[i] * inv_mass;
    float ay = fy[i] * inv_mass;
    float az = fz[i] * inv_mass;
    vx[i] += half_dt * ax;
    vy[i] += half_dt * ay;
    vz[i] += half_dt * az;
    px[i] += dt * vx[i];
    py[i] += dt * vy[i];
    pz[i] += dt * vz[i];
    vx[i] += half_dt * ax;
    vy[i] += half_dt * ay;
    vz[i] += half_dt * az;
  }
}
