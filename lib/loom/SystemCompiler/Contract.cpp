#include "loom/SystemCompiler/Contract.h"
#include "llvm/Support/JSON.h"

#include <algorithm>

using llvm::json::Array;
using llvm::json::Object;
using llvm::json::Value;

namespace loom {

//===----------------------------------------------------------------------===//
// Ordering enum conversion
//===----------------------------------------------------------------------===//

const char *orderingToString(Ordering o) {
  switch (o) {
  case Ordering::FIFO:
    return "FIFO";
  case Ordering::UNORDERED:
    return "UNORDERED";
  case Ordering::SYMBOLIC:
    return "SYMBOLIC";
  }
  return "FIFO";
}

Ordering orderingFromString(const std::string &s) {
  if (s == "UNORDERED")
    return Ordering::UNORDERED;
  if (s == "SYMBOLIC")
    return Ordering::SYMBOLIC;
  return Ordering::FIFO;
}

//===----------------------------------------------------------------------===//
// Placement enum conversion (new canonical name)
//===----------------------------------------------------------------------===//

const char *placementToString(Placement p) {
  switch (p) {
  case Placement::LOCAL_SPM:
    return "LOCAL_SPM";
  case Placement::SHARED_L2:
    return "SHARED_L2";
  case Placement::EXTERNAL:
    return "EXTERNAL";
  case Placement::AUTO:
    return "AUTO";
  }
  return "LOCAL_SPM";
}

Placement placementFromString(const std::string &s) {
  if (s == "SHARED_L2")
    return Placement::SHARED_L2;
  if (s == "EXTERNAL" || s == "EXTERNAL_DRAM")
    return Placement::EXTERNAL;
  if (s == "AUTO")
    return Placement::AUTO;
  return Placement::LOCAL_SPM;
}

//===----------------------------------------------------------------------===//
// Legacy visibility converters (delegate to placement)
//===----------------------------------------------------------------------===//

const char *visibilityToString(Visibility v) {
  return placementToString(v);
}

Visibility visibilityFromString(const std::string &s) {
  return placementFromString(s);
}

//===----------------------------------------------------------------------===//
// Backpressure enum conversion
//===----------------------------------------------------------------------===//

const char *backpressureToString(Backpressure b) {
  switch (b) {
  case Backpressure::BLOCK:
    return "BLOCK";
  case Backpressure::DROP:
    return "DROP";
  case Backpressure::OVERWRITE:
    return "OVERWRITE";
  }
  return "BLOCK";
}

Backpressure backpressureFromString(const std::string &s) {
  if (s == "DROP")
    return Backpressure::DROP;
  if (s == "OVERWRITE")
    return Backpressure::OVERWRITE;
  return Backpressure::BLOCK;
}

const char *writebackToString(Writeback w) {
  switch (w) {
  case Writeback::EAGER:
    return "EAGER";
  case Writeback::LAZY:
    return "LAZY";
  }
  return "EAGER";
}

Writeback writebackFromString(const std::string &s) {
  if (s == "LAZY")
    return Writeback::LAZY;
  return Writeback::EAGER;
}

const char *prefetchToString(Prefetch p) {
  switch (p) {
  case Prefetch::NONE:
    return "NONE";
  case Prefetch::NEXT_TILE:
    return "NEXT_TILE";
  case Prefetch::DOUBLE_BUFFER:
    return "DOUBLE_BUFFER";
  }
  return "NONE";
}

Prefetch prefetchFromString(const std::string &s) {
  if (s == "NEXT_TILE")
    return Prefetch::NEXT_TILE;
  if (s == "DOUBLE_BUFFER")
    return Prefetch::DOUBLE_BUFFER;
  return Prefetch::NONE;
}

Value contractSpecToJSON(const ContractSpec &spec) {
  Object obj;
  obj["producerKernel"] = spec.producerKernel;
  obj["consumerKernel"] = spec.consumerKernel;
  obj["dataTypeName"] = spec.dataTypeName;
  obj["ordering"] = orderingToString(spec.ordering);

  if (spec.productionRate)
    obj["productionRate"] = *spec.productionRate;
  if (spec.consumptionRate)
    obj["consumptionRate"] = *spec.consumptionRate;
  if (spec.steadyStateRatio) {
    Array ratio;
    ratio.push_back(spec.steadyStateRatio->first);
    ratio.push_back(spec.steadyStateRatio->second);
    obj["steadyStateRatio"] = std::move(ratio);
  }
  if (!spec.tileShape.empty()) {
    Array shape;
    for (int64_t dim : spec.tileShape)
      shape.push_back(dim);
    obj["tileShape"] = std::move(shape);
  }

  obj["minBufferElements"] = spec.minBufferElements;
  obj["maxBufferElements"] = spec.maxBufferElements;
  obj["backpressure"] = backpressureToString(spec.backpressure);
  obj["doubleBuffering"] = spec.doubleBuffering;

  obj["visibility"] = visibilityToString(spec.visibility);
  obj["producerWriteback"] = writebackToString(spec.producerWriteback);
  obj["consumerPrefetch"] = prefetchToString(spec.consumerPrefetch);

  obj["mayFuse"] = spec.mayFuse;
  obj["mayReplicate"] = spec.mayReplicate;
  obj["mayPipeline"] = spec.mayPipeline;
  obj["mayReorder"] = spec.mayReorder;
  obj["mayRetile"] = spec.mayRetile;

  if (spec.achievedProductionRate)
    obj["achievedProductionRate"] = *spec.achievedProductionRate;
  if (spec.achievedConsumptionRate)
    obj["achievedConsumptionRate"] = *spec.achievedConsumptionRate;
  if (spec.achievedBufferSize)
    obj["achievedBufferSize"] = *spec.achievedBufferSize;

  return Value(std::move(obj));
}

ContractSpec contractSpecFromJSON(const Value &v) {
  ContractSpec spec;
  auto *obj = v.getAsObject();
  if (!obj)
    return spec;

  if (auto s = obj->getString("producerKernel"))
    spec.producerKernel = s->str();
  if (auto s = obj->getString("consumerKernel"))
    spec.consumerKernel = s->str();
  if (auto s = obj->getString("dataTypeName"))
    spec.dataTypeName = s->str();
  if (auto s = obj->getString("ordering"))
    spec.ordering = orderingFromString(s->str());

  if (auto n = obj->getInteger("productionRate"))
    spec.productionRate = *n;
  if (auto n = obj->getInteger("consumptionRate"))
    spec.consumptionRate = *n;
  if (auto *arr = obj->getArray("steadyStateRatio")) {
    if (arr->size() == 2) {
      auto n0 = (*arr)[0].getAsInteger();
      auto n1 = (*arr)[1].getAsInteger();
      if (n0 && n1)
        spec.steadyStateRatio = {*n0, *n1};
    }
  }
  if (auto *arr = obj->getArray("tileShape")) {
    for (const auto &elem : *arr) {
      if (auto n = elem.getAsInteger())
        spec.tileShape.push_back(*n);
    }
  }

  if (auto n = obj->getInteger("minBufferElements"))
    spec.minBufferElements = *n;
  if (auto n = obj->getInteger("maxBufferElements"))
    spec.maxBufferElements = *n;
  if (auto s = obj->getString("backpressure"))
    spec.backpressure = backpressureFromString(s->str());
  if (auto b = obj->getBoolean("doubleBuffering"))
    spec.doubleBuffering = *b;

  if (auto s = obj->getString("visibility"))
    spec.visibility = visibilityFromString(s->str());
  if (auto s = obj->getString("producerWriteback"))
    spec.producerWriteback = writebackFromString(s->str());
  if (auto s = obj->getString("consumerPrefetch"))
    spec.consumerPrefetch = prefetchFromString(s->str());

  if (auto b = obj->getBoolean("mayFuse"))
    spec.mayFuse = *b;
  if (auto b = obj->getBoolean("mayReplicate"))
    spec.mayReplicate = *b;
  if (auto b = obj->getBoolean("mayPipeline"))
    spec.mayPipeline = *b;
  if (auto b = obj->getBoolean("mayReorder"))
    spec.mayReorder = *b;
  if (auto b = obj->getBoolean("mayRetile"))
    spec.mayRetile = *b;

  if (auto n = obj->getInteger("achievedProductionRate"))
    spec.achievedProductionRate = *n;
  if (auto n = obj->getInteger("achievedConsumptionRate"))
    spec.achievedConsumptionRate = *n;
  if (auto n = obj->getInteger("achievedBufferSize"))
    spec.achievedBufferSize = *n;

  return spec;
}

//===----------------------------------------------------------------------===//
// TDCEdgeSpec JSON serialization
//===----------------------------------------------------------------------===//

Value tdcEdgeSpecToJSON(const TDCEdgeSpec &spec) {
  Object obj;
  obj["producerKernel"] = spec.producerKernel;
  obj["consumerKernel"] = spec.consumerKernel;
  obj["dataTypeName"] = spec.dataTypeName;

  if (spec.ordering)
    obj["ordering"] = orderingToString(*spec.ordering);
  if (spec.throughput)
    obj["throughput"] = *spec.throughput;
  if (spec.placement)
    obj["placement"] = placementToString(*spec.placement);
  if (spec.shape)
    obj["shape"] = *spec.shape;

  return Value(std::move(obj));
}

TDCEdgeSpec tdcEdgeSpecFromJSON(const Value &v) {
  TDCEdgeSpec spec;
  auto *obj = v.getAsObject();
  if (!obj)
    return spec;

  if (auto s = obj->getString("producerKernel"))
    spec.producerKernel = s->str();
  if (auto s = obj->getString("consumerKernel"))
    spec.consumerKernel = s->str();
  if (auto s = obj->getString("dataTypeName"))
    spec.dataTypeName = s->str();

  if (auto s = obj->getString("ordering"))
    spec.ordering = orderingFromString(s->str());
  if (auto s = obj->getString("throughput"))
    spec.throughput = s->str();
  if (auto s = obj->getString("placement"))
    spec.placement = placementFromString(s->str());
  if (auto s = obj->getString("shape"))
    spec.shape = s->str();

  return spec;
}

//===----------------------------------------------------------------------===//
// TDCPathSpec JSON serialization
//===----------------------------------------------------------------------===//

Value tdcPathSpecToJSON(const TDCPathSpec &spec) {
  Object obj;
  obj["startProducer"] = spec.startProducer;
  obj["startConsumer"] = spec.startConsumer;
  obj["endProducer"] = spec.endProducer;
  obj["endConsumer"] = spec.endConsumer;
  obj["latency"] = spec.latency;
  return Value(std::move(obj));
}

TDCPathSpec tdcPathSpecFromJSON(const Value &v) {
  TDCPathSpec spec;
  auto *obj = v.getAsObject();
  if (!obj)
    return spec;

  if (auto s = obj->getString("startProducer"))
    spec.startProducer = s->str();
  if (auto s = obj->getString("startConsumer"))
    spec.startConsumer = s->str();
  if (auto s = obj->getString("endProducer"))
    spec.endProducer = s->str();
  if (auto s = obj->getString("endConsumer"))
    spec.endConsumer = s->str();
  if (auto s = obj->getString("latency"))
    spec.latency = s->str();

  return spec;
}

//===----------------------------------------------------------------------===//
// parseShapeExpr
//===----------------------------------------------------------------------===//

std::vector<std::string> parseShapeExpr(const std::string &shapeStr) {
  std::vector<std::string> dims;
  if (shapeStr.empty())
    return dims;

  // Strip surrounding brackets.
  std::string s = shapeStr;
  if (!s.empty() && s.front() == '[')
    s = s.substr(1);
  if (!s.empty() && s.back() == ']')
    s.pop_back();

  // Trim leading/trailing whitespace from the inner string.
  size_t start = s.find_first_not_of(' ');
  if (start == std::string::npos)
    return dims; // empty brackets "[]"
  size_t end = s.find_last_not_of(' ');
  s = s.substr(start, end - start + 1);

  if (s.empty())
    return dims;

  // Split by commas.
  size_t pos = 0;
  while (pos < s.size()) {
    size_t comma = s.find(',', pos);
    if (comma == std::string::npos)
      comma = s.size();
    std::string dim = s.substr(pos, comma - pos);
    // Trim whitespace.
    size_t dStart = dim.find_first_not_of(' ');
    size_t dEnd = dim.find_last_not_of(' ');
    if (dStart != std::string::npos)
      dims.push_back(dim.substr(dStart, dEnd - dStart + 1));
    pos = comma + 1;
  }

  return dims;
}

} // namespace loom
