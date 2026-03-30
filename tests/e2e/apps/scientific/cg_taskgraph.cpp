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

static void spmv_stub(const float *, const int32_t *, const float *, float *,
                      unsigned, unsigned) {}
static void dot_stub(const float *, const float *, float *, unsigned) {}
static void axpy_stub(float *, const float *, const float *, float, unsigned) {}
static void precond_stub(float *, const float *, const float *, unsigned,
                         unsigned) {}
static void convergence_stub(const float *, unsigned, float *) {}

static TaskGraph buildCGTaskGraph(const CGParams &params) {
  TaskGraph tg("conjugate_gradient");

  auto spmv = tg.kernel("spmv", spmv_stub);
  auto dot_pq = tg.kernel("dot_pq", dot_stub);
  auto axpy_x = tg.kernel("axpy_x", axpy_stub);
  auto axpy_r = tg.kernel("axpy_r", axpy_stub);
  auto precondition = tg.kernel("precondition", precond_stub);
  auto dot_rz = tg.kernel("dot_rz", dot_stub);
  auto axpy_p = tg.kernel("axpy_p", axpy_stub);
  auto convergence = tg.kernel("convergence_check", convergence_stub);

  tg.addVariant(spmv, "spmv_ell", VariantOptions{1, 1});
  tg.addVariant(spmv, "spmv_csr_unroll2", VariantOptions{2, 0});
  tg.addVariant(dot_pq, "dot_tree_reduce", VariantOptions{1, 1});
  tg.addVariant(dot_pq, "dot_sequential_unroll2", VariantOptions{2, 0});
  tg.addVariant(axpy_x, "axpy_unroll4", VariantOptions{4, 0});
  tg.addVariant(axpy_r, "axpy_unroll4", VariantOptions{4, 0});
  tg.addVariant(precondition, "precond_block4", VariantOptions{1, 1});
  tg.addVariant(dot_rz, "dot_tree_reduce", VariantOptions{1, 1});
  tg.addVariant(dot_rz, "dot_sequential_unroll2", VariantOptions{2, 0});
  tg.addVariant(axpy_p, "axpy_unroll4", VariantOptions{4, 0});
  tg.addVariant(convergence, "conv_max", VariantOptions{1, 1});

  const uint64_t vecBytes = cgVectorBytes(params);
  const uint64_t scalarBytes = cgScalarBytes();

  tg.connect(spmv, dot_pq)
      .ordering(Ordering::FIFO)
      .data_type<float>()
      .data_volume(2 * vecBytes)
      .placement(Placement::LOCAL_SPM);

  tg.connect(spmv, axpy_r)
      .ordering(Ordering::FIFO)
      .data_type<float>()
      .data_volume(vecBytes)
      .placement(Placement::LOCAL_SPM);

  tg.connect(dot_pq, axpy_x)
      .ordering(Ordering::FIFO)
      .data_type<float>()
      .data_volume(scalarBytes + vecBytes)
      .placement(Placement::LOCAL_SPM);

  tg.connect(dot_pq, axpy_r)
      .ordering(Ordering::FIFO)
      .data_type<float>()
      .data_volume(scalarBytes)
      .placement(Placement::LOCAL_SPM);

  tg.connect(axpy_r, precondition)
      .ordering(Ordering::FIFO)
      .data_type<float>()
      .data_volume(vecBytes)
      .placement(Placement::LOCAL_SPM);

  tg.connect(precondition, dot_rz)
      .ordering(Ordering::FIFO)
      .data_type<float>()
      .data_volume(2 * vecBytes)
      .placement(Placement::LOCAL_SPM);

  tg.connect(dot_rz, axpy_p)
      .ordering(Ordering::FIFO)
      .data_type<float>()
      .data_volume(scalarBytes + vecBytes)
      .placement(Placement::LOCAL_SPM);

  tg.connect(axpy_r, convergence)
      .ordering(Ordering::FIFO)
      .data_type<float>()
      .data_volume(vecBytes)
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

static llvm::cl::opt<unsigned> ProblemSize(
    "n", llvm::cl::desc("CG vector length"), llvm::cl::init(CGParams{}.n));

static llvm::cl::opt<unsigned> Nnz(
    "nnz", llvm::cl::desc("CG nonzero count"), llvm::cl::init(CGParams{}.nnz));

static llvm::cl::opt<unsigned> NnzPerRow(
    "nnz-per-row", llvm::cl::desc("CG nonzeros per row"),
    llvm::cl::init(CGParams{}.nnzPerRow));

static llvm::cl::opt<std::string> OutputPath(
    "o", llvm::cl::desc("Write TDG to file instead of stdout"),
    llvm::cl::init(""));

int main(int argc, char **argv) {
  llvm::InitLLVM initLLVM(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "Scientific CG TaskGraph\n");

  CGParams params;
  params.n = ProblemSize;
  params.nnz = Nnz;
  params.nnzPerRow = NnzPerRow;

  return emitGraph(buildCGTaskGraph(params), OutputPath);
}
