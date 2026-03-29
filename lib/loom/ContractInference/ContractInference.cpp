#include "loom/ContractInference/ContractInference.h"
#include "loom/ContractInference/BufferSizeInference.h"
#include "loom/ContractInference/RateAnalyzer.h"
#include "loom/ContractInference/TileShapeInference.h"
#include "loom/Dialect/TDG/TDGOps.h"
#include "loom/SystemCompiler/Contract.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/Support/raw_ostream.h"

#include <sstream>

using namespace loom;
using namespace mlir;

// Defined in VisibilityInference.cpp
namespace loom {
Visibility inferVisibility(int64_t productionRate, uint64_t tileElements,
                           unsigned elementSizeBytes,
                           uint64_t spmBudgetBytes, double spmThresholdFraction,
                           uint64_t l2BudgetBytes, double l2ThresholdFraction,
                           bool mayFuse);
} // namespace loom

/// Get the byte width of an MLIR type. Returns 0 for unknown types.
static unsigned getElementSizeBytes(Type type) {
  if (auto intType = dyn_cast<IntegerType>(type))
    return (intType.getWidth() + 7) / 8;
  if (auto floatType = dyn_cast<FloatType>(type))
    return floatType.getWidth() / 8;
  // Default: 4 bytes (assume 32-bit).
  return 4;
}

/// Find a kernel op by name within a graph op.
static tdg::KernelOp findKernel(tdg::GraphOp graphOp, StringRef name) {
  for (auto &op : graphOp.getBody().front()) {
    if (auto kernelOp = dyn_cast<tdg::KernelOp>(op)) {
      if (kernelOp.getSymName() == name)
        return kernelOp;
    }
  }
  return nullptr;
}

/// Process a single tdg.contract op: infer missing TDC dimensions.
///
/// The new TDC ContractOp has 4 optional string dimensions:
///   ordering, throughput, placement, tile_shape
///
/// This pass infers missing dimensions based on kernel body analysis and
/// memory hierarchy budgets. Legacy fields (backpressure, production_rate,
/// consumption_rate, etc.) are no longer present on the MLIR op and are
/// handled internally for inference purposes only.
static LogicalResult processContract(tdg::ContractOp contractOp,
                                     tdg::GraphOp graphOp,
                                     const ContractInferencePass::Options &opts) {
  OpBuilder builder(contractOp.getContext());
  unsigned elemSize = getElementSizeBytes(contractOp.getDataType());

  // Look up producer and consumer kernels.
  auto producerOp = findKernel(graphOp, contractOp.getProducer());
  auto consumerOp = findKernel(graphOp, contractOp.getConsumer());

  // --- Ordering inference ---
  // Default to FIFO if not set.
  if (!contractOp.getOrderingAttr()) {
    contractOp->setAttr("ordering", builder.getStringAttr("FIFO"));
  }

  // --- Rate analysis (used internally for tile shape and placement) ---
  RateAnalyzer rateAnalyzer;
  int64_t productionRate = 1;
  int64_t consumptionRate = 1;

  if (producerOp && !producerOp.getBody().empty()) {
    Value nullVal;
    auto rateResult =
        rateAnalyzer.analyzeProductionRate(producerOp.getBody(), nullVal);
    if (rateResult.elementsPerInvocation)
      productionRate = *rateResult.elementsPerInvocation;
  }

  if (consumerOp && !consumerOp.getBody().empty()) {
    Value nullVal;
    auto rateResult =
        rateAnalyzer.analyzeConsumptionRate(consumerOp.getBody(), nullVal);
    if (rateResult.elementsPerInvocation)
      consumptionRate = *rateResult.elementsPerInvocation;
  }

  // --- Tile shape inference ---
  TileShapeInference tileInfer;
  uint64_t tileElements = 1;
  std::vector<int64_t> tileShape;

  if (!contractOp.getTileShapeAttr()) {
    // Derive problem shape from producer tile_hint or rate.
    std::vector<int64_t> problemShape;
    if (producerOp && producerOp.getTileHint()) {
      auto hint = *producerOp.getTileHint();
      problemShape.assign(hint.begin(), hint.end());
    } else {
      problemShape = {productionRate};
    }

    auto tileResult = tileInfer.infer(problemShape, elemSize,
                                      opts.defaultSPMCapacityBytes);
    tileShape = tileResult.tileShape;
    tileElements = tileResult.elementsPerTile;

    // Serialize tile shape as a string expression "[d0, d1, ...]".
    if (!tileShape.empty()) {
      std::ostringstream oss;
      oss << "[";
      for (size_t idx = 0; idx < tileShape.size(); ++idx) {
        if (idx > 0)
          oss << ", ";
        oss << tileShape[idx];
      }
      oss << "]";
      contractOp->setAttr("tile_shape",
                          builder.getStringAttr(oss.str()));
    }
  } else {
    // Parse existing tile_shape string to compute tileElements.
    auto shapeStr = contractOp.getTileShapeAttr().getValue();
    auto dims = parseShapeExpr(shapeStr.str());
    tileElements = 1;
    for (const auto &dimStr : dims) {
      // Try to parse numeric dimensions; skip symbolic ones.
      char *end = nullptr;
      long val = std::strtol(dimStr.c_str(), &end, 10);
      if (end != dimStr.c_str() && *end == '\0' && val > 0) {
        tileElements *= static_cast<uint64_t>(val);
        tileShape.push_back(static_cast<int64_t>(val));
      }
    }
  }

  // --- Throughput inference ---
  // If throughput is not set, derive from production rate.
  if (!contractOp.getThroughputAttr() && productionRate > 1) {
    contractOp->setAttr("throughput",
                        builder.getStringAttr(std::to_string(productionRate)));
  }

  // --- Placement inference ---
  // Use VisibilityInference to determine memory placement if not set.
  // The mayFuse heuristic: producer and consumer are co-located if both
  // kernels exist (will be refined by the actual assignment later).
  bool mayFuse = (producerOp && consumerOp);

  if (!contractOp.getPlacementAttr()) {
    Placement inferred = inferVisibility(
        productionRate, tileElements, elemSize, opts.defaultSPMCapacityBytes,
        opts.spmThresholdFraction, opts.sharedL2CapacityBytes,
        opts.l2ThresholdFraction, mayFuse);

    contractOp->setAttr("placement",
                        builder.getStringAttr(placementToString(inferred)));
  }

  return success();
}

LogicalResult ContractInferencePass::run(ModuleOp tdgModule,
                                         const Options &opts) {
  bool hadError = false;

  // Walk all GraphOp instances within the module.
  tdgModule.walk([&](tdg::GraphOp graphOp) {
    // Walk all ContractOp instances within this graph.
    graphOp.walk([&](tdg::ContractOp contractOp) {
      if (failed(processContract(contractOp, graphOp, opts)))
        hadError = true;
    });
  });

  return hadError ? failure() : success();
}
