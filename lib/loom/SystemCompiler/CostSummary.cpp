#include "loom/SystemCompiler/CostSummary.h"
#include "llvm/Support/JSON.h"

using llvm::json::Array;
using llvm::json::Object;
using llvm::json::Value;

namespace loom {

Value kernelMetricsToJSON(const KernelMetrics &m) {
  Object obj;
  obj["kernelName"] = m.kernelName;
  obj["achievedII"] = static_cast<int64_t>(m.achievedII);
  obj["peUtilization"] = m.peUtilization;
  obj["fuUtilization"] = m.fuUtilization;
  obj["switchUtilization"] = m.switchUtilization;
  obj["spmBytesUsed"] = static_cast<int64_t>(m.spmBytesUsed);
  obj["achievedStreamRate"] = m.achievedStreamRate;
  return Value(std::move(obj));
}

KernelMetrics kernelMetricsFromJSON(const Value &v) {
  KernelMetrics m;
  auto *obj = v.getAsObject();
  if (!obj)
    return m;

  if (auto s = obj->getString("kernelName"))
    m.kernelName = s->str();
  if (auto n = obj->getInteger("achievedII"))
    m.achievedII = static_cast<unsigned>(*n);
  if (auto n = obj->getNumber("peUtilization"))
    m.peUtilization = *n;
  if (auto n = obj->getNumber("fuUtilization"))
    m.fuUtilization = *n;
  if (auto n = obj->getNumber("switchUtilization"))
    m.switchUtilization = *n;
  if (auto n = obj->getInteger("spmBytesUsed"))
    m.spmBytesUsed = static_cast<uint64_t>(*n);
  if (auto n = obj->getNumber("achievedStreamRate"))
    m.achievedStreamRate = *n;

  return m;
}

Value coreCostSummaryToJSON(const CoreCostSummary &s) {
  Object obj;
  obj["coreInstanceName"] = s.coreInstanceName;
  obj["coreType"] = s.coreType;
  obj["success"] = s.success;

  Array metrics;
  for (const auto &km : s.kernelMetrics)
    metrics.push_back(kernelMetricsToJSON(km));
  obj["kernelMetrics"] = std::move(metrics);

  obj["totalPEUtilization"] = s.totalPEUtilization;
  obj["totalSPMUtilization"] = s.totalSPMUtilization;
  obj["routingPressure"] = s.routingPressure;

  if (s.cut)
    obj["cut"] = infeasibilityCutToJSON(*s.cut);

  return Value(std::move(obj));
}

CoreCostSummary coreCostSummaryFromJSON(const Value &v) {
  CoreCostSummary s;
  auto *obj = v.getAsObject();
  if (!obj)
    return s;

  if (auto str = obj->getString("coreInstanceName"))
    s.coreInstanceName = str->str();
  if (auto str = obj->getString("coreType"))
    s.coreType = str->str();
  if (auto b = obj->getBoolean("success"))
    s.success = *b;

  if (auto *arr = obj->getArray("kernelMetrics")) {
    for (const auto &elem : *arr)
      s.kernelMetrics.push_back(kernelMetricsFromJSON(elem));
  }

  if (auto n = obj->getNumber("totalPEUtilization"))
    s.totalPEUtilization = *n;
  if (auto n = obj->getNumber("totalSPMUtilization"))
    s.totalSPMUtilization = *n;
  if (auto n = obj->getNumber("routingPressure"))
    s.routingPressure = *n;

  if (auto *cutObj = obj->get("cut"))
    s.cut = infeasibilityCutFromJSON(*cutObj);

  return s;
}

} // namespace loom
