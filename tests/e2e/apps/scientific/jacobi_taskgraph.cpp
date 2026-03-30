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

static void halo_exchange_stub(float *, const float *, float *, unsigned,
                               unsigned, unsigned) {}
static void stencil_compute_stub(const float *, float *, unsigned, unsigned,
                                 unsigned) {}
static void boundary_apply_stub(float *, unsigned, unsigned, float) {}
static void residual_check_stub(const float *, const float *, unsigned,
                                unsigned) {}

static TaskGraph buildJacobiTaskGraph(const JacobiParams &params) {
  TaskGraph tg("jacobi_2d");

  auto halo = tg.kernel("halo_exchange", halo_exchange_stub);
  auto stencil = tg.kernel("stencil_compute", stencil_compute_stub);
  auto boundary = tg.kernel("boundary_apply", boundary_apply_stub);
  auto residual = tg.kernel("residual_check", residual_check_stub);

  tg.addVariant(halo, "halo_row_col", VariantOptions{1, 1});
  tg.addVariant(stencil, "stencil_9pt", VariantOptions{1, 1});
  tg.addVariant(stencil, "stencil_5pt_unroll2", VariantOptions{2, 0});
  tg.addVariant(boundary, "boundary_neumann", VariantOptions{1, 1});
  tg.addVariant(residual, "residual_max", VariantOptions{1, 1});

  tg.connect(halo, stencil)
      .ordering(Ordering::FIFO)
      .data_type<float>()
      .data_volume(haloBytes(params))
      .placement(Placement::LOCAL_SPM);

  tg.connect(stencil, boundary)
      .ordering(Ordering::FIFO)
      .data_type<float>()
      .data_volume(tileBytes(params))
      .placement(Placement::LOCAL_SPM);

  tg.connect(boundary, residual)
      .ordering(Ordering::FIFO)
      .data_type<float>()
      .data_volume(2 * tileBytes(params))
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

static llvm::cl::opt<unsigned> TileRows(
    "tile-rows", llvm::cl::desc("Jacobi tile rows"),
    llvm::cl::init(JacobiParams{}.tileRows));

static llvm::cl::opt<unsigned> TileCols(
    "tile-cols", llvm::cl::desc("Jacobi tile cols"),
    llvm::cl::init(JacobiParams{}.tileCols));

static llvm::cl::opt<unsigned> HaloWidth(
    "halo-w", llvm::cl::desc("Jacobi halo width"),
    llvm::cl::init(JacobiParams{}.haloWidth));

static llvm::cl::opt<std::string> OutputPath(
    "o", llvm::cl::desc("Write TDG to file instead of stdout"),
    llvm::cl::init(""));

int main(int argc, char **argv) {
  llvm::InitLLVM initLLVM(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv, "Scientific Jacobi TaskGraph\n");

  JacobiParams params;
  params.tileRows = TileRows;
  params.tileCols = TileCols;
  params.haloWidth = HaloWidth;

  return emitGraph(buildJacobiTaskGraph(params), OutputPath);
}
