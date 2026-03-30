#include "nbody_ref.h"

#include <algorithm>
#include <cmath>

namespace scicomp_test {

namespace {

static void zero_forces(NBodyState &state) {
  std::fill(state.fx.begin(), state.fx.end(), 0.0f);
  std::fill(state.fy.begin(), state.fy.end(), 0.0f);
  std::fill(state.fz.begin(), state.fz.end(), 0.0f);
}

static void accumulate_pair(int i, int j, const NBodyState &state, float g,
                            float softening, std::vector<float> &fx,
                            std::vector<float> &fy, std::vector<float> &fz) {
  const float dx = state.px[static_cast<size_t>(j)] -
                   state.px[static_cast<size_t>(i)];
  const float dy = state.py[static_cast<size_t>(j)] -
                   state.py[static_cast<size_t>(i)];
  const float dz = state.pz[static_cast<size_t>(j)] -
                   state.pz[static_cast<size_t>(i)];
  const float dist2 = dx * dx + dy * dy + dz * dz + softening * softening;
  const float inv_dist = 1.0f / std::sqrt(dist2);
  const float inv_dist3 = inv_dist * inv_dist * inv_dist;
  const float scale = g * state.mass[static_cast<size_t>(i)] *
                      state.mass[static_cast<size_t>(j)] * inv_dist3;
  const float fx_ij = dx * scale;
  const float fy_ij = dy * scale;
  const float fz_ij = dz * scale;
  fx[static_cast<size_t>(i)] += fx_ij;
  fy[static_cast<size_t>(i)] += fy_ij;
  fz[static_cast<size_t>(i)] += fz_ij;
  fx[static_cast<size_t>(j)] -= fx_ij;
  fy[static_cast<size_t>(j)] -= fy_ij;
  fz[static_cast<size_t>(j)] -= fz_ij;
}

static void compute_direct_forces(NBodyState &state, float g,
                                  float softening) {
  zero_forces(state);
  const int n = state.size();
  for (int i = 0; i < n; ++i) {
    for (int j = i + 1; j < n; ++j)
      accumulate_pair(i, j, state, g, softening, state.fx, state.fy,
                      state.fz);
  }
}

} // namespace

float compute_nbody_energy(const NBodyState &state, float g, float softening) {
  const int n = state.size();
  if (n <= 0)
    return 0.0f;

  float total = 0.0f;
  for (int i = 0; i < n; ++i) {
    const float vx = state.vx[static_cast<size_t>(i)];
    const float vy = state.vy[static_cast<size_t>(i)];
    const float vz = state.vz[static_cast<size_t>(i)];
    const float mass = state.mass[static_cast<size_t>(i)];
    total += 0.5f * mass * (vx * vx + vy * vy + vz * vz);
  }

  const float soft2 = softening * softening;
  for (int i = 0; i < n; ++i) {
    for (int j = i + 1; j < n; ++j) {
      const float dx = state.px[static_cast<size_t>(j)] -
                       state.px[static_cast<size_t>(i)];
      const float dy = state.py[static_cast<size_t>(j)] -
                       state.py[static_cast<size_t>(i)];
      const float dz = state.pz[static_cast<size_t>(j)] -
                       state.pz[static_cast<size_t>(i)];
      const float dist = std::sqrt(dx * dx + dy * dy + dz * dz + soft2);
      if (dist > 0.0f)
        total -= g * state.mass[static_cast<size_t>(i)] *
                 state.mass[static_cast<size_t>(j)] / dist;
    }
  }

  return total;
}

NBodyReferenceResult run_nbody_reference(const NBodyState &initial, int steps,
                                         float dt, float g, float softening) {
  NBodyReferenceResult result;
  result.state = initial;
  const int n = result.state.size();
  if (n <= 0 || steps < 0)
    return result;

  result.energy_history.reserve(static_cast<size_t>(steps) + 1);
  result.energy_history.push_back(
      compute_nbody_energy(result.state, g, softening));

  for (int step = 0; step < steps; ++step) {
    compute_direct_forces(result.state, g, softening);
    const float half_dt = 0.5f * dt;
    for (int i = 0; i < n; ++i) {
      const float inv_mass = result.state.mass[static_cast<size_t>(i)] != 0.0f
                                 ? 1.0f / result.state.mass[static_cast<size_t>(i)]
                                 : 1.0f;
      const float ax = result.state.fx[static_cast<size_t>(i)] * inv_mass;
      const float ay = result.state.fy[static_cast<size_t>(i)] * inv_mass;
      const float az = result.state.fz[static_cast<size_t>(i)] * inv_mass;
      result.state.vx[static_cast<size_t>(i)] += half_dt * ax;
      result.state.vy[static_cast<size_t>(i)] += half_dt * ay;
      result.state.vz[static_cast<size_t>(i)] += half_dt * az;
      result.state.px[static_cast<size_t>(i)] +=
          dt * result.state.vx[static_cast<size_t>(i)];
      result.state.py[static_cast<size_t>(i)] +=
          dt * result.state.vy[static_cast<size_t>(i)];
      result.state.pz[static_cast<size_t>(i)] +=
          dt * result.state.vz[static_cast<size_t>(i)];
      result.state.vx[static_cast<size_t>(i)] += half_dt * ax;
      result.state.vy[static_cast<size_t>(i)] += half_dt * ay;
      result.state.vz[static_cast<size_t>(i)] += half_dt * az;
    }
    result.energy_history.push_back(
        compute_nbody_energy(result.state, g, softening));
  }

  return result;
}

} // namespace scicomp_test
