#include "reference/nbody_ref.h"
#include "scicomp_params.h"
#include "test_scicomp_utils.h"

#include "tapestry/task_graph.h"
#include "tapestry/tdg_emitter.h"

#include "loom/Dialect/TDG/TDGDialect.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"

#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <vector>

extern "C" {
void force_direct(const float *px, const float *py, const float *pz,
                  const float *mass, float *fx, float *fy, float *fz, int n,
                  float g, float softening);
void force_cutoff(const float *px, const float *py, const float *pz,
                  const float *mass, const int *offsets, const int *indices,
                  float *fx, float *fy, float *fz, int n, float g,
                  float softening);
void force_tree(const float *px, const float *py, const float *pz,
                const float *mass, const int *tree_offsets,
                const int *tree_indices, float *fx, float *fy, float *fz,
                int n, float g, float softening);
void force_direct_unroll2(const float *px, const float *py, const float *pz,
                          const float *mass, float *fx, float *fy, float *fz,
                          int n, float g, float softening);
void update_verlet(float *px, float *py, float *pz, float *vx, float *vy,
                   float *vz, const float *fx, const float *fy,
                   const float *fz, const float *mass, int n, float dt);
void update_verlet_unroll2(float *px, float *py, float *pz, float *vx,
                           float *vy, float *vz, const float *fx,
                           const float *fy, const float *fz,
                           const float *mass, int n, float dt);
int rebuild_cell_list(const float *px, const float *py, const float *pz, int n,
                      float cutoff, int *offsets, int *indices,
                      int max_neighbors);
int rebuild_verlet_list(const float *px, const float *py, const float *pz,
                        int n, float cutoff, float skin, int *offsets,
                        int *indices, int max_neighbors);
float energy_ke_only(const float *vx, const float *vy, const float *vz,
                     const float *mass, int n);
float energy_ke_pe(const float *px, const float *py, const float *pz,
                   const float *vx, const float *vy, const float *vz,
                   const float *mass, const int *offsets, const int *indices,
                   int n, float g, float softening);
}

namespace {

using namespace scicomp;
using namespace scicomp_test;

static void force_stub(const float *, const float *, const float *, float *,
                       unsigned) {}
static void update_stub(float *, float *, const float *, const float *,
                        float, unsigned) {}
static void rebuild_stub(const float *, uint32_t *, unsigned, unsigned,
                         unsigned) {}
static void energy_stub(const float *, const float *, const float *,
                        const float *, float *, unsigned) {}

static tapestry::TaskGraph buildNBodyTaskGraph(const NBodyParams &params) {
  tapestry::TaskGraph tg("nbody_simulation");

  auto force = tg.kernel("force_compute", force_stub);
  auto update = tg.kernel("position_update", update_stub);
  auto rebuild = tg.kernel("neighbor_rebuild", rebuild_stub);
  auto energy = tg.kernel("energy_reduce", energy_stub);

  tg.addVariant(force, "force_cutoff", tapestry::VariantOptions{1, 1});
  tg.addVariant(force, "force_tree", tapestry::VariantOptions{1, 2});
  tg.addVariant(force, "force_direct_unroll2",
                tapestry::VariantOptions{2, 0});
  tg.addVariant(update, "update_verlet_unroll2",
                tapestry::VariantOptions{2, 0});
  tg.addVariant(rebuild, "rebuild_verlet_list",
                tapestry::VariantOptions{1, 1});
  tg.addVariant(energy, "energy_ke_pe", tapestry::VariantOptions{1, 1});

  const uint64_t particle_bytes = nbodyParticleBytes(params);
  const uint64_t neighbor_bytes = nbodyNeighborBytes(params);

  tg.connect(force, update)
      .ordering(tapestry::Ordering::FIFO)
      .data_type<float>()
      .data_volume(static_cast<uint64_t>(params.nParticles) * 3 * fp32Bytes)
      .placement(tapestry::Placement::LOCAL_SPM);
  tg.connect(update, rebuild)
      .ordering(tapestry::Ordering::FIFO)
      .data_type<float>()
      .data_volume(static_cast<uint64_t>(params.nParticles) * 3 * fp32Bytes)
      .placement(tapestry::Placement::LOCAL_SPM);
  tg.connect(update, energy)
      .ordering(tapestry::Ordering::FIFO)
      .data_type<float>()
      .data_volume(particle_bytes)
      .placement(tapestry::Placement::LOCAL_SPM);
  tg.connect(rebuild, force)
      .ordering(tapestry::Ordering::FIFO)
      .data_type<int32_t>()
      .data_volume(neighbor_bytes)
      .placement(tapestry::Placement::LOCAL_SPM);

  return tg;
}

static bool emit_nbody_tdg() {
  mlir::MLIRContext ctx;
  ctx.getOrLoadDialect<loom::tdg::TDGDialect>();
  NBodyParams params;
  params.nParticles = 8;
  auto module = tapestry::emitTDG(buildNBodyTaskGraph(params), ctx);
  if (!module) {
    std::puts("FAIL nbody tdg emission");
    return false;
  }
  return true;
}

static float manual_kinetic_energy(const NBodyState &state) {
  float total = 0.0f;
  for (int i = 0; i < state.size(); ++i) {
    const float vx = state.vx[static_cast<size_t>(i)];
    const float vy = state.vy[static_cast<size_t>(i)];
    const float vz = state.vz[static_cast<size_t>(i)];
    const float mass = state.mass[static_cast<size_t>(i)];
    total += 0.5f * mass * (vx * vx + vy * vy + vz * vz);
  }
  return total;
}

static std::vector<float> pack_forces(const std::vector<float> &fx,
                                      const std::vector<float> &fy,
                                      const std::vector<float> &fz) {
  std::vector<float> packed;
  packed.reserve(fx.size() * 3);
  for (size_t i = 0; i < fx.size(); ++i) {
    packed.push_back(fx[i]);
    packed.push_back(fy[i]);
    packed.push_back(fz[i]);
  }
  return packed;
}

static bool compare_state_vectors(const NBodyState &lhs, const NBodyState &rhs,
                                  float eps) {
  return vectors_close(lhs.px, rhs.px, eps, eps) &&
         vectors_close(lhs.py, rhs.py, eps, eps) &&
         vectors_close(lhs.pz, rhs.pz, eps, eps) &&
         vectors_close(lhs.vx, rhs.vx, eps, eps) &&
         vectors_close(lhs.vy, rhs.vy, eps, eps) &&
         vectors_close(lhs.vz, rhs.vz, eps, eps);
}

static bool check_force_variants(const NBodyState &state, float g,
                                 float softening,
                                 const std::vector<int> &offsets,
                                 const std::vector<int> &indices) {
  std::vector<float> fx(static_cast<size_t>(state.size()), 0.0f);
  std::vector<float> fy(static_cast<size_t>(state.size()), 0.0f);
  std::vector<float> fz(static_cast<size_t>(state.size()), 0.0f);
  std::vector<float> fx_unroll = fx;
  std::vector<float> fy_unroll = fy;
  std::vector<float> fz_unroll = fz;
  std::vector<float> fx_cutoff = fx;
  std::vector<float> fy_cutoff = fy;
  std::vector<float> fz_cutoff = fz;

  force_direct(state.px.data(), state.py.data(), state.pz.data(),
               state.mass.data(), fx.data(), fy.data(), fz.data(), state.size(),
               g, softening);
  force_direct_unroll2(state.px.data(), state.py.data(), state.pz.data(),
                       state.mass.data(), fx_unroll.data(), fy_unroll.data(),
                       fz_unroll.data(), state.size(), g, softening);
  force_cutoff(state.px.data(), state.py.data(), state.pz.data(),
               state.mass.data(), offsets.data(), indices.data(),
               fx_cutoff.data(), fy_cutoff.data(), fz_cutoff.data(),
               state.size(), g, softening);

  if (!vectors_close(fx, fx_unroll, 1e-6f, 1e-6f) ||
      !vectors_close(fy, fy_unroll, 1e-6f, 1e-6f) ||
      !vectors_close(fz, fz_unroll, 1e-6f, 1e-6f)) {
    std::puts("FAIL nbody force_direct vs force_direct_unroll2");
    return false;
  }
  if (!vectors_close(fx, fx_cutoff, 1e-6f, 1e-6f) ||
      !vectors_close(fy, fy_cutoff, 1e-6f, 1e-6f) ||
      !vectors_close(fz, fz_cutoff, 1e-6f, 1e-6f)) {
    std::puts("FAIL nbody force_direct vs force_cutoff");
    return false;
  }
  return true;
}

static bool check_update_variants(const NBodyState &state, float dt) {
  NBodyState lhs = state;
  NBodyState rhs = state;
  force_direct(lhs.px.data(), lhs.py.data(), lhs.pz.data(), lhs.mass.data(),
               lhs.fx.data(), lhs.fy.data(), lhs.fz.data(), lhs.size(), 0.05f,
               0.1f);
  force_direct_unroll2(rhs.px.data(), rhs.py.data(), rhs.pz.data(),
                       rhs.mass.data(), rhs.fx.data(), rhs.fy.data(),
                       rhs.fz.data(), rhs.size(), 0.05f, 0.1f);
  update_verlet(lhs.px.data(), lhs.py.data(), lhs.pz.data(), lhs.vx.data(),
                lhs.vy.data(), lhs.vz.data(), lhs.fx.data(), lhs.fy.data(),
                lhs.fz.data(), lhs.mass.data(), lhs.size(), dt);
  update_verlet_unroll2(rhs.px.data(), rhs.py.data(), rhs.pz.data(),
                        rhs.vx.data(), rhs.vy.data(), rhs.vz.data(),
                        rhs.fx.data(), rhs.fy.data(), rhs.fz.data(),
                        rhs.mass.data(), rhs.size(), dt);
  if (!compare_state_vectors(lhs, rhs, 1e-6f)) {
    std::puts("FAIL nbody update_verlet vs update_verlet_unroll2");
    return false;
  }
  return true;
}

static bool run_nbody_case() {
  const int n = 8;
  const float g = 0.05f;
  const float softening = 0.1f;
  const float dt = 0.0005f;
  const int steps = 10;

  NBodyState initial = make_nbody_state(n, 42, 0.6f);
  const int max_neighbors = n * (n - 1);
  std::vector<int> offsets(static_cast<size_t>(n) + 1, 0);
  std::vector<int> indices(static_cast<size_t>(max_neighbors), 0);
  std::vector<int> offsets2 = offsets;
  std::vector<int> indices2 = indices;

  const int rebuilt = rebuild_cell_list(initial.px.data(), initial.py.data(),
                                        initial.pz.data(), n, 1000.0f,
                                        offsets.data(), indices.data(),
                                        max_neighbors);
  const int rebuilt2 = rebuild_verlet_list(initial.px.data(), initial.py.data(),
                                           initial.pz.data(), n, 1000.0f, 0.0f,
                                           offsets2.data(), indices2.data(),
                                           max_neighbors);
  if (rebuilt != max_neighbors || rebuilt2 != max_neighbors) {
    std::printf("FAIL nbody rebuild counts %d %d\n", rebuilt, rebuilt2);
    return false;
  }
  if (offsets != offsets2 || indices != indices2) {
    std::puts("FAIL nbody rebuild variants differ");
    return false;
  }

  if (!check_force_variants(initial, g, softening, offsets, indices))
    return false;
  if (!check_update_variants(initial, dt))
    return false;

  std::vector<float> fx(static_cast<size_t>(n), 0.0f);
  std::vector<float> fy(static_cast<size_t>(n), 0.0f);
  std::vector<float> fz(static_cast<size_t>(n), 0.0f);
  std::vector<float> energy_history;
  energy_history.reserve(static_cast<size_t>(steps) + 1);

  NBodyState run = initial;
  energy_history.push_back(compute_nbody_energy(run, g, softening));
  for (int step = 0; step < steps; ++step) {
    force_direct(run.px.data(), run.py.data(), run.pz.data(), run.mass.data(),
                 fx.data(), fy.data(), fz.data(), n, g, softening);
    update_verlet(run.px.data(), run.py.data(), run.pz.data(), run.vx.data(),
                  run.vy.data(), run.vz.data(), fx.data(), fy.data(),
                  fz.data(), run.mass.data(), n, dt);
    run.fx = fx;
    run.fy = fy;
    run.fz = fz;
    energy_history.push_back(energy_ke_pe(run.px.data(), run.py.data(),
                                          run.pz.data(), run.vx.data(),
                                          run.vy.data(), run.vz.data(),
                                          run.mass.data(), offsets.data(),
                                          indices.data(), n, g, softening));
  }

  const NBodyReferenceResult ref =
      run_nbody_reference(initial, steps, dt, g, softening);
  if (!compare_state_vectors(run, ref.state, 1e-3f)) {
    std::puts("FAIL nbody reference state mismatch");
    return false;
  }
  if (!vectors_close(energy_history, ref.energy_history, 1e-4f, 1e-4f)) {
    std::puts("FAIL nbody energy history mismatch");
    return false;
  }

  const float initial_energy = std::fabs(energy_history.front());
  const float final_energy = std::fabs(energy_history.back());
  const float drift =
      std::fabs(final_energy - initial_energy) / std::max(initial_energy, 1e-6f);
  if (drift > 0.01f) {
    std::printf("FAIL nbody energy drift %f\n", drift);
    return false;
  }

  const float ke_kernel = energy_ke_only(run.vx.data(), run.vy.data(),
                                         run.vz.data(), run.mass.data(), n);
  if (!nearly_equal(ke_kernel, manual_kinetic_energy(run), 1e-5f, 1e-5f)) {
    std::puts("FAIL nbody kinetic energy mismatch");
    return false;
  }

  const float energy_kernel =
      energy_ke_pe(run.px.data(), run.py.data(), run.pz.data(), run.vx.data(),
                   run.vy.data(), run.vz.data(), run.mass.data(), offsets.data(),
                   indices.data(), n, g, softening);
  if (!nearly_equal(energy_kernel, energy_history.back(), 1e-4f, 1e-4f)) {
    std::puts("FAIL nbody total energy mismatch");
    return false;
  }

  const auto packed_forces = pack_forces(run.fx, run.fy, run.fz);
  (void)packed_forces;
  return true;
}

} // namespace

int main(int argc, char **argv) {
  llvm::InitLLVM initLLVM(argc, argv);
  (void)initLLVM;

  bool ok = emit_nbody_tdg();
  ok = ok && run_nbody_case();

  std::puts(ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
