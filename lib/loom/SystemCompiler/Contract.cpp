#include "loom/SystemCompiler/Contract.h"
#include "llvm/Support/JSON.h"

using llvm::json::Array;
using llvm::json::Object;
using llvm::json::Value;

namespace loom {

const char *orderingToString(Ordering o) {
  switch (o) {
  case Ordering::FIFO:
    return "FIFO";
  case Ordering::UNORDERED:
    return "UNORDERED";
  }
  return "FIFO";
}

Ordering orderingFromString(const std::string &s) {
  if (s == "UNORDERED")
    return Ordering::UNORDERED;
  return Ordering::FIFO;
}

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

const char *visibilityToString(Visibility v) {
  switch (v) {
  case Visibility::LOCAL_SPM:
    return "LOCAL_SPM";
  case Visibility::SHARED_L2:
    return "SHARED_L2";
  case Visibility::EXTERNAL_DRAM:
    return "EXTERNAL_DRAM";
  }
  return "LOCAL_SPM";
}

Visibility visibilityFromString(const std::string &s) {
  if (s == "SHARED_L2")
    return Visibility::SHARED_L2;
  if (s == "EXTERNAL_DRAM")
    return Visibility::EXTERNAL_DRAM;
  return Visibility::LOCAL_SPM;
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

} // namespace loom
