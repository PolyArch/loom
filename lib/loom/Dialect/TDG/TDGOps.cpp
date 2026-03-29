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
  // Validate ordering value when present
  if (auto ord = getOrdering()) {
    if (*ord != "FIFO" && *ord != "UNORDERED" && *ord != "SYMBOLIC") {
      return emitOpError() << "invalid ordering '" << *ord
                           << "'; expected FIFO, UNORDERED, or SYMBOLIC";
    }
  }

  // Validate placement value when present
  if (auto plc = getPlacement()) {
    if (*plc != "LOCAL_SPM" && *plc != "SHARED_L2" && *plc != "EXTERNAL" &&
        *plc != "AUTO") {
      return emitOpError()
             << "invalid placement '" << *plc
             << "'; expected LOCAL_SPM, SHARED_L2, EXTERNAL, or AUTO";
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

// Print: tdg.contract @producer -> @consumer {data_type = f32, ...}
void ContractOp::print(OpAsmPrinter &p) {
  p << " " << getProducerAttr() << " -> " << getConsumerAttr();
  p << " {";
  p << "data_type = " << getDataType();

  if (auto ord = getOrderingAttr()) {
    p << ", ordering = ";
    p.printKeywordOrString(ord.getValue());
  }
  if (auto thr = getThroughputAttr()) {
    p << ", throughput = ";
    p.printKeywordOrString(thr.getValue());
  }
  if (auto plc = getPlacementAttr()) {
    p << ", placement = ";
    p.printKeywordOrString(plc.getValue());
  }
  if (auto ts = getTileShapeAttr()) {
    p << ", tile_shape = ";
    p.printKeywordOrString(ts.getValue());
  }

  p << "}";

  // Print remaining attributes not covered above
  SmallVector<StringRef> elidedAttrs = {"producer",   "consumer",  "data_type",
                                        "ordering",   "throughput", "placement",
                                        "tile_shape"};
  p.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

// Parse: tdg.contract @producer -> @consumer {data_type = f32, ...}
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

  // Mandatory: data_type
  if (parser.parseKeyword("data_type") || parser.parseEqual())
    return failure();
  TypeAttr dataTypeAttr;
  if (parser.parseAttribute(dataTypeAttr, "data_type", result.attributes))
    return failure();

  auto &builder = parser.getBuilder();

  // Parse optional key-value pairs
  while (succeeded(parser.parseOptionalComma())) {
    std::string key;
    if (parser.parseKeywordOrString(&key) || parser.parseEqual())
      return failure();

    if (key == "ordering" || key == "throughput" || key == "placement" ||
        key == "tile_shape") {
      std::string val;
      if (parser.parseKeywordOrString(&val))
        return failure();
      result.addAttribute(key, builder.getStringAttr(val));
    } else {
      return parser.emitError(parser.getCurrentLocation(),
                              "unknown contract attribute '")
             << key << "'";
    }
  }

  if (parser.parseRBrace())
    return failure();

  // Parse optional remaining attributes
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  return success();
}

//===----------------------------------------------------------------------===//
// PathContractOp verifier
//===----------------------------------------------------------------------===//

LogicalResult PathContractOp::verify() {
  // Validate that start_edge and end_edge reference existing ContractOp
  // symbols within the parent GraphOp. Since ContractOp does not have a
  // SymbolName, we look for ContractOps whose producer->consumer pair
  // matches the referenced name. The reference format is the edge name
  // as a flat symbol ref.
  //
  // For now we just validate that the parent graph exists and the latency
  // is non-empty. Full symbol resolution requires naming contracts.
  if (getLatency().empty()) {
    return emitOpError() << "latency expression must not be empty";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// PathContractOp custom assembly format
//===----------------------------------------------------------------------===//

// Print: tdg.path_contract @edge0 -> @edge1 {latency = "4 * tile_m"}
void PathContractOp::print(OpAsmPrinter &p) {
  p << " " << getStartEdgeAttr() << " -> " << getEndEdgeAttr();
  p << " {latency = ";
  p.printKeywordOrString(getLatency());
  p << "}";

  SmallVector<StringRef> elidedAttrs = {"start_edge", "end_edge", "latency"};
  p.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

// Parse: tdg.path_contract @edge0 -> @edge1 {latency = "4 * tile_m"}
ParseResult PathContractOp::parse(OpAsmParser &parser, OperationState &result) {
  FlatSymbolRefAttr startEdgeAttr, endEdgeAttr;
  if (parser.parseAttribute(startEdgeAttr, "start_edge", result.attributes))
    return failure();
  if (parser.parseArrow())
    return failure();
  if (parser.parseAttribute(endEdgeAttr, "end_edge", result.attributes))
    return failure();

  if (parser.parseLBrace())
    return failure();

  if (parser.parseKeyword("latency") || parser.parseEqual())
    return failure();

  std::string latencyStr;
  if (parser.parseKeywordOrString(&latencyStr))
    return failure();
  result.addAttribute("latency",
                       parser.getBuilder().getStringAttr(latencyStr));

  if (parser.parseRBrace())
    return failure();

  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  return success();
}

#define GET_OP_CLASSES
#include "loom/Dialect/TDG/TDGOps.cpp.inc"
