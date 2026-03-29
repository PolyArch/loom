#include "tapestry/tdg_emitter.h"
#include "tapestry/task_graph.h"

#include "loom/Dialect/TDG/TDGDialect.h"
#include "loom/Dialect/TDG/TDGOps.h"
#include "loom/Dialect/TDG/TDGTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/Support/raw_ostream.h"

#include <fstream>

using namespace mlir;

namespace tapestry {

// ---------------------------------------------------------------------------
// Helper: resolve data-type name to MLIR Type.
// ---------------------------------------------------------------------------
static Type resolveDataType(OpBuilder &builder, const std::string &name) {
  if (name == "f32")
    return builder.getF32Type();
  if (name == "f64")
    return builder.getF64Type();
  if (name == "f16")
    return builder.getF16Type();
  if (name == "i32")
    return builder.getI32Type();
  if (name == "i64")
    return builder.getI64Type();
  if (name == "i16")
    return builder.getI16Type();
  if (name == "i8" || name == "u8")
    return builder.getIntegerType(8);
  if (name == "u16")
    return builder.getIntegerType(16, /*isSigned=*/false);
  if (name == "u32")
    return builder.getIntegerType(32, /*isSigned=*/false);
  if (name == "u64")
    return builder.getIntegerType(64, /*isSigned=*/false);
  // Fallback to f32.
  return builder.getF32Type();
}

// ---------------------------------------------------------------------------
// Helper: convert tapestry::Ordering to loom ordering string.
// ---------------------------------------------------------------------------
static StringRef orderingStr(std::optional<Ordering> o) {
  if (!o)
    return "FIFO"; // default
  switch (*o) {
  case Ordering::FIFO:
    return "FIFO";
  case Ordering::UNORDERED:
    return "UNORDERED";
  }
  return "FIFO";
}

// ---------------------------------------------------------------------------
// Helper: convert tapestry::Placement to string.
// ---------------------------------------------------------------------------
static StringRef placementStr(std::optional<Placement> p) {
  if (!p)
    return "AUTO"; // default
  switch (*p) {
  case Placement::LOCAL_SPM:
    return "LOCAL_SPM";
  case Placement::SHARED_L2:
    return "SHARED_L2";
  case Placement::EXTERNAL:
    return "EXTERNAL";
  case Placement::AUTO:
    return "AUTO";
  }
  return "AUTO";
}

// ---------------------------------------------------------------------------
// emitTDG
// ---------------------------------------------------------------------------

OwningOpRef<ModuleOp> emitTDG(const TaskGraph &graph, MLIRContext &ctx) {
  // Load TDG dialect.
  ctx.getOrLoadDialect<loom::tdg::TDGDialect>();

  auto loc = UnknownLoc::get(&ctx);
  OwningOpRef<ModuleOp> module = ModuleOp::create(loc);
  OpBuilder builder(&ctx);
  builder.setInsertionPointToEnd(module->getBody());

  // Create tdg.graph.
  auto graphOp = loom::tdg::GraphOp::create(
      builder, loc,
      /*sym_name=*/builder.getStringAttr(graph.name()),
      /*target_arch=*/StringAttr());

  // Create the single block in the graph body.
  Block *graphBlock = new Block();
  graphOp.getBody().push_back(graphBlock);
  builder.setInsertionPointToEnd(graphBlock);

  // Emit tdg.kernel ops (one per kernel).
  const auto &kernels = graph.kernels();
  for (const auto &ki : kernels) {
    // Determine kernel_type from execution target.
    std::string kernelType;
    switch (ki.target) {
    case ExecutionTarget::HOST:
      kernelType = "host";
      break;
    case ExecutionTarget::CGRA:
      kernelType = "cgra";
      break;
    case ExecutionTarget::AUTO_DETECT:
    default:
      kernelType = "auto";
      break;
    }

    auto kernelOp = loom::tdg::KernelOp::create(
        builder, loc,
        /*sym_name=*/builder.getStringAttr(ki.name),
        /*kernel_type=*/builder.getStringAttr(kernelType),
        /*tile_hint=*/DenseI64ArrayAttr(),
        /*target_core_type=*/StringAttr());

    // Create an empty kernel body (kernel_compiler from C08 fills it later).
    Block *kernelBlock = new Block();
    kernelOp.getBody().push_back(kernelBlock);
  }

  // Emit tdg.contract ops (one per edge).
  graph.forEachEdge([&](const std::string &producerName,
                        const std::string &consumerName,
                        const Contract &contract) {
    // Fill defaults before emission.
    std::string dataTypeName =
        contract.dataTypeName.value_or("f32");
    Type dataType = resolveDataType(builder, dataTypeName);

    // Map tapestry::Placement to the TDG dialect visibility string.
    StringRef visibilityStr = "LOCAL_SPM";
    if (contract.placement) {
      switch (*contract.placement) {
      case Placement::LOCAL_SPM:
        visibilityStr = "LOCAL_SPM";
        break;
      case Placement::SHARED_L2:
        visibilityStr = "SHARED_L2";
        break;
      case Placement::EXTERNAL:
        visibilityStr = "EXTERNAL_DRAM";
        break;
      case Placement::AUTO:
        visibilityStr = "LOCAL_SPM";
        break;
      }
    }

    loom::tdg::ContractOp::create(
        builder, loc,
        /*producer=*/producerName,
        /*consumer=*/consumerName,
        /*ordering=*/orderingStr(contract.ordering),
        /*data_type=*/dataType,
        /*production_rate=*/AffineMapAttr(),
        /*consumption_rate=*/AffineMapAttr(),
        /*steady_state_ratio=*/DenseI64ArrayAttr(),
        /*tile_shape=*/DenseI64ArrayAttr(),
        /*min_buffer_elements=*/IntegerAttr(),
        /*max_buffer_elements=*/IntegerAttr(),
        /*backpressure=*/StringRef("BLOCK"),
        /*double_buffering=*/false,
        /*visibility=*/visibilityStr,
        /*producer_writeback=*/StringRef("EAGER"),
        /*consumer_prefetch=*/StringRef("NONE"),
        /*may_fuse=*/true,
        /*may_replicate=*/true,
        /*may_pipeline=*/true,
        /*may_reorder=*/false,
        /*may_retile=*/true);
  });

  return module;
}

// ---------------------------------------------------------------------------
// writeTDGToFile
// ---------------------------------------------------------------------------

bool writeTDGToFile(ModuleOp module, const std::string &path) {
  std::error_code ec;
  llvm::raw_fd_ostream output(path, ec);
  if (ec)
    return false;
  module.print(output);
  return true;
}

} // namespace tapestry
