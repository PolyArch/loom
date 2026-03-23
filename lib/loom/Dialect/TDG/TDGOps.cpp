#include "loom/Dialect/TDG/TDGOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/StringSet.h"

using namespace mlir;
using namespace loom::tdg;

//===----------------------------------------------------------------------===//
// GraphOp verifier
//===----------------------------------------------------------------------===//

LogicalResult GraphOp::verify() {
  llvm::StringSet<> kernelNames;
  for (auto &op : getBody().front()) {
    if (auto kernelOp = dyn_cast<KernelOp>(op)) {
      StringRef name = kernelOp.getSymName();
      if (!kernelNames.insert(name).second) {
        return kernelOp.emitOpError()
               << "duplicate kernel name '" << name << "' in graph";
      }
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// ContractOp verifier
//===----------------------------------------------------------------------===//

LogicalResult ContractOp::verify() {
  // Check ordering/reorder consistency: FIFO + may_reorder=true is invalid
  StringRef ord = getOrdering();
  if (ord == "FIFO" && getMayReorder()) {
    return emitOpError()
           << "FIFO ordering is incompatible with may_reorder=true";
  }

  // Validate ordering value
  if (ord != "FIFO" && ord != "UNORDERED" && ord != "AFFINE_INDEXED") {
    return emitOpError() << "invalid ordering '" << ord
                         << "'; expected FIFO, UNORDERED, or AFFINE_INDEXED";
  }

  // Validate backpressure value
  StringRef bp = getBackpressure();
  if (bp != "BLOCK" && bp != "DROP" && bp != "OVERWRITE") {
    return emitOpError() << "invalid backpressure '" << bp
                         << "'; expected BLOCK, DROP, or OVERWRITE";
  }

  // Validate visibility value
  StringRef vis = getVisibility();
  if (vis != "LOCAL_SPM" && vis != "SHARED_L2" && vis != "EXTERNAL_DRAM") {
    return emitOpError()
           << "invalid visibility '" << vis
           << "'; expected LOCAL_SPM, SHARED_L2, or EXTERNAL_DRAM";
  }

  // Validate writeback value
  StringRef wb = getProducerWriteback();
  if (wb != "EAGER" && wb != "LAZY") {
    return emitOpError() << "invalid producer_writeback '" << wb
                         << "'; expected EAGER or LAZY";
  }

  // Validate prefetch value
  StringRef pf = getConsumerPrefetch();
  if (pf != "NONE" && pf != "NEXT_TILE" && pf != "DOUBLE_BUFFER") {
    return emitOpError()
           << "invalid consumer_prefetch '" << pf
           << "'; expected NONE, NEXT_TILE, or DOUBLE_BUFFER";
  }

  // Validate steady_state_ratio has exactly 2 elements if present
  if (auto ratio = getSteadyStateRatio()) {
    if (ratio->size() != 2) {
      return emitOpError()
             << "steady_state_ratio must have exactly 2 elements "
             << "(numerator, denominator), got " << ratio->size();
    }
  }

  // Check that referenced producer and consumer kernels exist in parent graph
  auto graphOp = (*this)->getParentOfType<GraphOp>();
  if (graphOp) {
    StringRef producerName = getProducer();
    StringRef consumerName = getConsumer();
    bool foundProducer = false;
    bool foundConsumer = false;
    for (auto &op : graphOp.getBody().front()) {
      if (auto kernelOp = dyn_cast<KernelOp>(op)) {
        if (kernelOp.getSymName() == producerName)
          foundProducer = true;
        if (kernelOp.getSymName() == consumerName)
          foundConsumer = true;
      }
    }
    if (!foundProducer) {
      return emitOpError() << "references unknown producer kernel '"
                           << producerName << "'";
    }
    if (!foundConsumer) {
      return emitOpError() << "references unknown consumer kernel '"
                           << consumerName << "'";
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ContractOp custom assembly format
//===----------------------------------------------------------------------===//

// Print: tdg.contract @producer -> @consumer {ordering = FIFO, ...}
void ContractOp::print(OpAsmPrinter &p) {
  p << " " << getProducerAttr() << " -> " << getConsumerAttr();
  p << " {";
  p << "ordering = " << getOrdering();
  p << ", data_type = " << getDataType();

  if (auto rate = getProductionRateAttr())
    p << ", production_rate = " << rate;
  if (auto rate = getConsumptionRateAttr())
    p << ", consumption_rate = " << rate;
  if (auto ratio = getSteadyStateRatioAttr())
    p << ", steady_state_ratio = " << ratio;
  if (auto shape = getTileShapeAttr())
    p << ", tile_shape = " << shape;
  if (auto minBuf = getMinBufferElements())
    p << ", min_buffer_elements = " << *minBuf;
  if (auto maxBuf = getMaxBufferElements())
    p << ", max_buffer_elements = " << *maxBuf;

  // Only print non-default values
  if (getBackpressure() != "BLOCK")
    p << ", backpressure = " << getBackpressure();
  if (getDoubleBuffering())
    p << ", double_buffering = true";
  if (getVisibility() != "LOCAL_SPM")
    p << ", visibility = " << getVisibility();
  if (getProducerWriteback() != "EAGER")
    p << ", producer_writeback = " << getProducerWriteback();
  if (getConsumerPrefetch() != "NONE")
    p << ", consumer_prefetch = " << getConsumerPrefetch();
  if (!getMayFuse())
    p << ", may_fuse = false";
  if (!getMayReplicate())
    p << ", may_replicate = false";
  if (!getMayPipeline())
    p << ", may_pipeline = false";
  if (getMayReorder())
    p << ", may_reorder = true";
  if (!getMayRetile())
    p << ", may_retile = false";

  p << "}";

  // Print remaining attributes not covered above
  SmallVector<StringRef> elidedAttrs = {
      "producer",          "consumer",         "ordering",
      "data_type",         "production_rate",  "consumption_rate",
      "steady_state_ratio", "tile_shape",      "min_buffer_elements",
      "max_buffer_elements", "backpressure",   "double_buffering",
      "visibility",        "producer_writeback", "consumer_prefetch",
      "may_fuse",          "may_replicate",    "may_pipeline",
      "may_reorder",       "may_retile"};
  p.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

// Parse: tdg.contract @producer -> @consumer {ordering = FIFO, ...}
ParseResult ContractOp::parse(OpAsmParser &parser, OperationState &result) {
  FlatSymbolRefAttr producerAttr, consumerAttr;
  if (parser.parseAttribute(producerAttr, "producer", result.attributes))
    return failure();
  if (parser.parseArrow())
    return failure();
  if (parser.parseAttribute(consumerAttr, "consumer", result.attributes))
    return failure();

  // Parse the brace-enclosed attribute list
  if (parser.parseLBrace())
    return failure();

  // Mandatory: ordering
  StringAttr orderingAttr;
  if (parser.parseKeyword("ordering") || parser.parseEqual())
    return failure();
  std::string orderingStr;
  if (parser.parseKeywordOrString(&orderingStr))
    return failure();
  orderingAttr = parser.getBuilder().getStringAttr(orderingStr);
  result.addAttribute("ordering", orderingAttr);

  // Mandatory: data_type
  if (parser.parseComma() || parser.parseKeyword("data_type") ||
      parser.parseEqual())
    return failure();
  TypeAttr dataTypeAttr;
  if (parser.parseAttribute(dataTypeAttr, "data_type", result.attributes))
    return failure();

  // Set defaults for optional attributes
  auto &builder = parser.getBuilder();
  result.addAttribute("backpressure", builder.getStringAttr("BLOCK"));
  result.addAttribute("double_buffering", builder.getBoolAttr(false));
  result.addAttribute("visibility", builder.getStringAttr("LOCAL_SPM"));
  result.addAttribute("producer_writeback", builder.getStringAttr("EAGER"));
  result.addAttribute("consumer_prefetch", builder.getStringAttr("NONE"));
  result.addAttribute("may_fuse", builder.getBoolAttr(true));
  result.addAttribute("may_replicate", builder.getBoolAttr(true));
  result.addAttribute("may_pipeline", builder.getBoolAttr(true));
  result.addAttribute("may_reorder", builder.getBoolAttr(false));
  result.addAttribute("may_retile", builder.getBoolAttr(true));

  // Parse optional key-value pairs
  while (succeeded(parser.parseOptionalComma())) {
    std::string key;
    if (parser.parseKeywordOrString(&key) || parser.parseEqual())
      return failure();

    if (key == "production_rate" || key == "consumption_rate") {
      AffineMapAttr mapAttr;
      if (parser.parseAttribute(mapAttr, key, result.attributes))
        return failure();
    } else if (key == "steady_state_ratio" || key == "tile_shape") {
      // Remove any existing attr with same name (won't be set yet for these)
      DenseI64ArrayAttr arrayAttr;
      if (parser.parseAttribute(arrayAttr, key, result.attributes))
        return failure();
    } else if (key == "min_buffer_elements" || key == "max_buffer_elements") {
      IntegerAttr intAttr;
      if (parser.parseAttribute(intAttr,
                                builder.getIntegerType(64), key,
                                result.attributes))
        return failure();
    } else if (key == "backpressure" || key == "visibility" ||
               key == "producer_writeback" || key == "consumer_prefetch") {
      // Override default
      std::string val;
      if (parser.parseKeywordOrString(&val))
        return failure();
      // Remove existing default and set new value
      result.attributes.set(key, builder.getStringAttr(val));
    } else if (key == "double_buffering" || key == "may_fuse" ||
               key == "may_replicate" || key == "may_pipeline" ||
               key == "may_reorder" || key == "may_retile") {
      bool val;
      if (parser.parseKeywordOrString(&orderingStr))
        return failure();
      val = (orderingStr == "true");
      result.attributes.set(key, builder.getBoolAttr(val));
    } else {
      return parser.emitError(parser.getCurrentLocation(),
                              "unknown contract attribute '") << key << "'";
    }
  }

  if (parser.parseRBrace())
    return failure();

  // Parse optional remaining attributes
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  return success();
}

#define GET_OP_CLASSES
#include "loom/Dialect/TDG/TDGOps.cpp.inc"
