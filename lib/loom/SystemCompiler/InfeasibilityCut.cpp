#include "loom/SystemCompiler/InfeasibilityCut.h"
#include "llvm/Support/JSON.h"

using llvm::json::Array;
using llvm::json::Object;
using llvm::json::Value;

namespace loom {

const char *cutReasonToString(CutReason r) {
  switch (r) {
  case CutReason::INSUFFICIENT_FU:
    return "INSUFFICIENT_FU";
  case CutReason::ROUTING_CONGESTION:
    return "ROUTING_CONGESTION";
  case CutReason::SPM_OVERFLOW:
    return "SPM_OVERFLOW";
  case CutReason::II_UNACHIEVABLE:
    return "II_UNACHIEVABLE";
  case CutReason::TYPE_MISMATCH:
    return "TYPE_MISMATCH";
  }
  return "INSUFFICIENT_FU";
}

CutReason cutReasonFromString(const std::string &s) {
  if (s == "ROUTING_CONGESTION")
    return CutReason::ROUTING_CONGESTION;
  if (s == "SPM_OVERFLOW")
    return CutReason::SPM_OVERFLOW;
  if (s == "II_UNACHIEVABLE")
    return CutReason::II_UNACHIEVABLE;
  if (s == "TYPE_MISMATCH")
    return CutReason::TYPE_MISMATCH;
  return CutReason::INSUFFICIENT_FU;
}

Value infeasibilityCutToJSON(const InfeasibilityCut &cut) {
  Object obj;
  obj["kernelName"] = cut.kernelName;
  obj["coreType"] = cut.coreType;
  obj["reason"] = cutReasonToString(cut.reason);

  std::visit(
      [&obj](auto &&evidence) {
        using T = std::decay_t<decltype(evidence)>;
        if constexpr (std::is_same_v<T, FUShortage>) {
          Object ev;
          ev["fuType"] = evidence.fuType;
          ev["needed"] = static_cast<int64_t>(evidence.needed);
          ev["available"] = static_cast<int64_t>(evidence.available);
          obj["fuShortage"] = std::move(ev);
        } else if constexpr (std::is_same_v<T, CongestionInfo>) {
          Object ev;
          ev["utilizationPct"] = evidence.utilizationPct;
          obj["congestionInfo"] = std::move(ev);
        } else if constexpr (std::is_same_v<T, SPMInfo>) {
          Object ev;
          ev["neededBytes"] = static_cast<int64_t>(evidence.neededBytes);
          ev["availableBytes"] = static_cast<int64_t>(evidence.availableBytes);
          obj["spmInfo"] = std::move(ev);
        } else if constexpr (std::is_same_v<T, IIInfo>) {
          Object ev;
          ev["minII"] = static_cast<int64_t>(evidence.minII);
          ev["targetII"] = static_cast<int64_t>(evidence.targetII);
          obj["iiInfo"] = std::move(ev);
        }
      },
      cut.evidence);

  return Value(std::move(obj));
}

InfeasibilityCut infeasibilityCutFromJSON(const Value &v) {
  InfeasibilityCut cut;
  auto *obj = v.getAsObject();
  if (!obj)
    return cut;

  if (auto s = obj->getString("kernelName"))
    cut.kernelName = s->str();
  if (auto s = obj->getString("coreType"))
    cut.coreType = s->str();
  if (auto s = obj->getString("reason"))
    cut.reason = cutReasonFromString(s->str());

  if (auto *ev = obj->getObject("fuShortage")) {
    FUShortage shortage;
    if (auto s = ev->getString("fuType"))
      shortage.fuType = s->str();
    if (auto n = ev->getInteger("needed"))
      shortage.needed = static_cast<unsigned>(*n);
    if (auto n = ev->getInteger("available"))
      shortage.available = static_cast<unsigned>(*n);
    cut.evidence = shortage;
  } else if (auto *ev = obj->getObject("congestionInfo")) {
    CongestionInfo info;
    if (auto n = ev->getNumber("utilizationPct"))
      info.utilizationPct = *n;
    cut.evidence = info;
  } else if (auto *ev = obj->getObject("spmInfo")) {
    SPMInfo info;
    if (auto n = ev->getInteger("neededBytes"))
      info.neededBytes = static_cast<uint64_t>(*n);
    if (auto n = ev->getInteger("availableBytes"))
      info.availableBytes = static_cast<uint64_t>(*n);
    cut.evidence = info;
  } else if (auto *ev = obj->getObject("iiInfo")) {
    IIInfo info;
    if (auto n = ev->getInteger("minII"))
      info.minII = static_cast<unsigned>(*n);
    if (auto n = ev->getInteger("targetII"))
      info.targetII = static_cast<unsigned>(*n);
    cut.evidence = info;
  }

  return cut;
}

} // namespace loom
