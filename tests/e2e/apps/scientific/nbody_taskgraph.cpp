#include "scicomp_params.h"

#include "tapestry/task_graph.h"
#include "tapestry/tdg_emitter.h"

#include "loom/Dialect/TDG/TDGDialect.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

using namespace tapestry;
using namespace scicomp;

namespace {

static void force_direct_stub(const float *, const float *, const float *,
                              float *, unsigned) {}
static void position_update_stub(float *, float *, const float *, const float *,
                                 float, unsigned) {}
static void neighbor_rebuild_stub(const float *, uint32_t *, unsigned,
                                  unsigned, unsigned) {}
static void energy_reduce_stub(const float *, const float *, const float *,
                               const float *, float *, unsigned) {}

static TaskGraph buildNBodyTaskGraph(const NBodyParams &params) {
  TaskGraph tg("nbody_simulation");

  auto force = tg.kernel("force_compute", force_direct_stub);
  auto update = tg.kernel("position_update", position_update_stub);
  auto rebuild = tg.kernel("neighbor_rebuild", neighbor_rebuild_stub);
  auto energy = tg.kernel("energy_reduce", energy_reduce_stub);

  tg.addVariant(force, "force_cutoff", VariantOptions{1, 1});
  tg.addVariant(force, "force_tree", VariantOptions{1, 2});
  tg.addVariant(force, "force_direct_unroll2", VariantOptions{2, 0});
  tg.addVariant(update, "update_verlet_unroll2", VariantOptions{2, 0});
  tg.addVariant(rebuild, "rebuild_verlet_list", VariantOptions{1, 1});
  tg.addVariant(energy, "energy_ke_pe", VariantOptions{1, 1});

  const uint64_t particleBytes = nbodyParticleBytes(params);
  const uint64_t neighborBytes = nbodyNeighborBytes(params);

  tg.connect(force, update)
      .ordering(Ordering::FIFO)
      .data_type<float>()
      .data_volume(static_cast<uint64_t>(params.nParticles) * 3 * fp32Bytes)
      .placement(Placement::LOCAL_SPM);

  tg.connect(update, rebuild)
      .ordering(Ordering::FIFO)
      .data_type<float>()
      .data_volume(static_cast<uint64_t>(params.nParticles) * 3 * fp32Bytes)
      .placement(Placement::LOCAL_SPM);

  tg.connect(update, energy)
      .ordering(Ordering::FIFO)
      .data_type<float>()
      .data_volume(particleBytes)
      .placement(Placement::LOCAL_SPM);

  tg.connect(rebuild, force)
      .ordering(Ordering::FIFO)
      .data_type<int32_t>()
      .data_volume(neighborBytes)
      .placement(Placement::LOCAL_SPM);

  return tg;
}

static int emitGraph(const TaskGraph &tg, const std::string &outputPath) {
  mlir::MLIRContext ctx;
  ctx.getOrLoadDialect<loom::tdg::TDGDialect>();

  auto module = emitTDG(tg, ctx);
  if (!module) {
    llvm::errs() << "failed to emit TDG\n";
    return 1;
  }

  if (outputPath.empty()) {
    module->print(llvm::outs());
    llvm::outs() << '\n';
    return 0;
  }

  if (!writeTDGToFile(*module, outputPath)) {
    llvm::errs() << "failed to write TDG to '" << outputPath << "'\n";
    return 1;
  }

  return 0;
}

} // namespace

static llvm::cl::opt<unsigned> ParticleCount(
    "n-particles", llvm::cl::desc("N-body particle count"),
    llvm::cl::init(NBodyParams{}.nParticles));

static llvm::cl::opt<unsigned> NeighborCount(
    "n-neighbors", llvm::cl::desc("N-body neighbor count"),
    llvm::cl::init(NBodyParams{}.nNeighbors));

static llvm::cl::opt<unsigned> RebuildInterval(
    "rebuild-interval", llvm::cl::desc("N-body neighbor rebuild interval"),
    llvm::cl::init(NBodyParams{}.rebuildInterval));

static llvm::cl::opt<std::string> OutputPath(
    "o", llvm::cl::desc("Write TDG to file instead of stdout"),
    llvm::cl::init(""));

int main(int argc, char **argv) {
  llvm::InitLLVM initLLVM(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "Scientific N-Body TaskGraph\n");

  NBodyParams params;
  params.nParticles = ParticleCount;
  params.nNeighbors = NeighborCount;
  params.rebuildInterval = RebuildInterval;

  return emitGraph(buildNBodyTaskGraph(params), OutputPath);
}
