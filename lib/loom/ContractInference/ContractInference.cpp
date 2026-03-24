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

/// Process a single tdg.contract op: infer missing fields.
static LogicalResult processContract(tdg::ContractOp contractOp,
                                     tdg::GraphOp graphOp,
                                     const ContractInferencePass::Options &opts) {
  OpBuilder builder(contractOp.getContext());
  unsigned elemSize = getElementSizeBytes(contractOp.getDataType());

  // Warn if DROP or OVERWRITE backpressure is specified -- these modes are
  // reserved for future implementation. Fall back to BLOCK behavior.
  StringRef bp = contractOp.getBackpressure();
  if (bp == "DROP" || bp == "OVERWRITE") {
    contractOp.emitWarning()
        << bp
        << " backpressure not yet implemented, falling back to BLOCK";
    contractOp->setAttr("backpressure", builder.getStringAttr("BLOCK"));
  }

  // Look up producer and consumer kernels.
  auto producerOp = findKernel(graphOp, contractOp.getProducer());
  auto consumerOp = findKernel(graphOp, contractOp.getConsumer());

  // RateAnalyzer: infer production and consumption rates if not set.
  RateAnalyzer rateAnalyzer;

  int64_t productionRate = 1;
  int64_t consumptionRate = 1;

  if (!contractOp.getProductionRateAttr() && producerOp &&
      !producerOp.getBody().empty()) {
    // Analyze producer kernel body.
    // Use a null Value since we analyze the entire kernel body.
    Value nullVal;
    auto rateResult =
        rateAnalyzer.analyzeProductionRate(producerOp.getBody(), nullVal);
    if (rateResult.elementsPerInvocation)
      productionRate = *rateResult.elementsPerInvocation;
  } else if (contractOp.getProductionRateAttr()) {
    // The production rate is stored as an AffineMap attribute.
    // For constant maps, extract the single result.
    auto map = contractOp.getProductionRateAttr().getValue();
    if (map.isConstant())
      productionRate = map.getSingleConstantResult();
  }

  if (!contractOp.getConsumptionRateAttr() && consumerOp &&
      !consumerOp.getBody().empty()) {
    Value nullVal;
    auto rateResult =
        rateAnalyzer.analyzeConsumptionRate(consumerOp.getBody(), nullVal);
    if (rateResult.elementsPerInvocation)
      consumptionRate = *rateResult.elementsPerInvocation;
  } else if (contractOp.getConsumptionRateAttr()) {
    auto map = contractOp.getConsumptionRateAttr().getValue();
    if (map.isConstant())
      consumptionRate = map.getSingleConstantResult();
  }

  // TileShapeInference: infer tile shape if not set.
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
      // Use production rate as a 1D problem shape.
      problemShape = {productionRate};
    }

    auto tileResult = tileInfer.infer(problemShape, elemSize,
                                      opts.defaultSPMCapacityBytes);
    tileShape = tileResult.tileShape;
    tileElements = tileResult.elementsPerTile;

    // Set the tile_shape attribute on the contract.
    if (!tileShape.empty()) {
      contractOp->setAttr("tile_shape",
                          builder.getDenseI64ArrayAttr(tileShape));
    }
  } else {
    auto existingShapeAttr = contractOp.getTileShapeAttr();
    tileShape.assign(existingShapeAttr.asArrayRef().begin(),
                     existingShapeAttr.asArrayRef().end());
    tileElements = 1;
    for (int64_t dim : tileShape)
      tileElements *= static_cast<uint64_t>(dim);
  }

  // Compute steady_state_ratio if not set.
  if (!contractOp.getSteadyStateRatioAttr() && consumptionRate > 0) {
    // Simplify ratio by GCD.
    int64_t num = productionRate;
    int64_t den = consumptionRate;
    auto gcd = [](int64_t a, int64_t b) -> int64_t {
      while (b) {
        a %= b;
        std::swap(a, b);
      }
      return a;
    };
    int64_t g = gcd(num > 0 ? num : -num, den > 0 ? den : -den);
    if (g > 0) {
      num /= g;
      den /= g;
    }
    contractOp->setAttr("steady_state_ratio",
                        builder.getDenseI64ArrayAttr({num, den}));
  }

  // BufferSizeInference: infer buffer sizes if not set.
  if (!contractOp.getMinBufferElements() ||
      !contractOp.getMaxBufferElements()) {
    ContractSpec spec;
    spec.ordering = orderingFromString(
        contractOp.getOrdering().str());
    spec.productionRate = productionRate;
    spec.consumptionRate = consumptionRate;
    spec.tileShape = tileShape;

    BufferSizeInference bufInfer;
    auto bufResult = bufInfer.infer(spec, opts.defaultSPMCapacityBytes,
                                    elemSize,
                                    opts.defaultProducerLatencyCycles);

    if (!contractOp.getMinBufferElements()) {
      contractOp->setAttr("min_buffer_elements",
                          builder.getI64IntegerAttr(bufResult.minElements));
    }
    if (!contractOp.getMaxBufferElements()) {
      contractOp->setAttr("max_buffer_elements",
                          builder.getI64IntegerAttr(bufResult.maxElements));
    }

    // Set double_buffering if the inference recommends it.
    if (bufResult.requiresDoubleBuffering && !contractOp.getDoubleBuffering()) {
      contractOp->setAttr("double_buffering", builder.getBoolAttr(true));
    }
  }

  // VisibilityInference: infer visibility if still at default.
  // Only override if the user has not explicitly set it (we check if it
  // is the default LOCAL_SPM -- if the user wanted LOCAL_SPM they would
  // leave it unset, so this is safe to override).
  Visibility inferred = inferVisibility(
      productionRate, tileElements, elemSize, opts.defaultSPMCapacityBytes,
      opts.spmThresholdFraction, opts.sharedL2CapacityBytes,
      opts.l2ThresholdFraction, contractOp.getMayFuse());

  contractOp->setAttr("visibility",
                       builder.getStringAttr(visibilityToString(inferred)));

  // Set may_reorder based on ordering.
  Ordering ordering = orderingFromString(contractOp.getOrdering().str());
  if (ordering == Ordering::UNORDERED) {
    contractOp->setAttr("may_reorder", builder.getBoolAttr(true));
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
