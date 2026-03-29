#include "loom/SystemCompiler/AssignmentPlan.h"
#include "llvm/Support/JSON.h"

using llvm::json::Array;
using llvm::json::Object;
using llvm::json::Value;

namespace loom {

Value AssignmentPlan::toJSON() const {
  Object obj;

  // kernelToCore.
  Object k2cObj;
  for (const auto &entry : kernelToCore)
    k2cObj[entry.first] = static_cast<int64_t>(entry.second);
  obj["kernelToCore"] = std::move(k2cObj);

  // coreAssignments.
  Array caArr;
  for (const auto &ca : coreAssignments) {
    Object caObj;
    caObj["coreInstanceIdx"] = static_cast<int64_t>(ca.coreInstanceIdx);
    caObj["coreTypeName"] = ca.coreTypeName;
    Array kernelArr;
    for (const auto &k : ca.assignedKernels)
      kernelArr.push_back(k);
    caObj["assignedKernels"] = std::move(kernelArr);
    caObj["estimatedUtilization"] = ca.estimatedUtilization;
    caArr.push_back(std::move(caObj));
  }
  obj["coreAssignments"] = std::move(caArr);

  // schedulingOrder.
  Array soArr;
  for (const auto &s : schedulingOrder)
    soArr.push_back(s);
  obj["schedulingOrder"] = std::move(soArr);

  // nocPaths.
  Array npArr;
  for (const auto &route : nocPaths) {
    Object routeObj;
    routeObj["contractEdgeName"] = route.contractEdgeName;
    routeObj["producerCore"] = route.producerCore;
    routeObj["consumerCore"] = route.consumerCore;
    routeObj["numHops"] = static_cast<int64_t>(route.numHops);
    routeObj["bandwidthFlitsPerCycle"] =
        static_cast<int64_t>(route.bandwidthFlitsPerCycle);
    routeObj["transferLatencyCycles"] =
        static_cast<int64_t>(route.transferLatencyCycles);

    Array hopsArr;
    for (const auto &hop : route.hops) {
      Array hopPair;
      hopPair.push_back(static_cast<int64_t>(hop.first));
      hopPair.push_back(static_cast<int64_t>(hop.second));
      hopsArr.push_back(std::move(hopPair));
    }
    routeObj["hops"] = std::move(hopsArr);
    npArr.push_back(std::move(routeObj));
  }
  obj["nocPaths"] = std::move(npArr);

  // objectiveValue.
  Object ovObj;
  ovObj["latency"] = objectiveValue.latency;
  ovObj["nocCost"] = objectiveValue.nocCost;
  ovObj["localityBonus"] = objectiveValue.localityBonus;
  obj["objectiveValue"] = std::move(ovObj);

  return Value(std::move(obj));
}

AssignmentPlan AssignmentPlan::fromJSON(const Value &v) {
  AssignmentPlan plan;
  auto *obj = v.getAsObject();
  if (!obj)
    return plan;

  // kernelToCore.
  if (auto *k2cObj = obj->getObject("kernelToCore")) {
    for (const auto &entry : *k2cObj) {
      if (auto n = entry.second.getAsInteger())
        plan.kernelToCore[entry.first.str()] = static_cast<unsigned>(*n);
    }
  }

  // coreAssignments.
  if (auto *caArr = obj->getArray("coreAssignments")) {
    for (const auto &entry : *caArr) {
      auto *caObj = entry.getAsObject();
      if (!caObj)
        continue;
      CoreAssignment ca;
      if (auto n = caObj->getInteger("coreInstanceIdx"))
        ca.coreInstanceIdx = static_cast<unsigned>(*n);
      if (auto s = caObj->getString("coreTypeName"))
        ca.coreTypeName = s->str();
      if (auto *kernelArr = caObj->getArray("assignedKernels")) {
        for (const auto &k : *kernelArr) {
          if (auto s = k.getAsString())
            ca.assignedKernels.push_back(s->str());
        }
      }
      if (auto n = caObj->getNumber("estimatedUtilization"))
        ca.estimatedUtilization = *n;
      plan.coreAssignments.push_back(std::move(ca));
    }
  }

  // schedulingOrder.
  if (auto *soArr = obj->getArray("schedulingOrder")) {
    for (const auto &entry : *soArr) {
      if (auto s = entry.getAsString())
        plan.schedulingOrder.push_back(s->str());
    }
  }

  // nocPaths.
  if (auto *npArr = obj->getArray("nocPaths")) {
    for (const auto &entry : *npArr) {
      auto *routeObj = entry.getAsObject();
      if (!routeObj)
        continue;
      NoCRoute route;
      if (auto s = routeObj->getString("contractEdgeName"))
        route.contractEdgeName = s->str();
      if (auto s = routeObj->getString("producerCore"))
        route.producerCore = s->str();
      if (auto s = routeObj->getString("consumerCore"))
        route.consumerCore = s->str();
      if (auto n = routeObj->getInteger("numHops"))
        route.numHops = static_cast<unsigned>(*n);
      if (auto n = routeObj->getInteger("bandwidthFlitsPerCycle"))
        route.bandwidthFlitsPerCycle = static_cast<unsigned>(*n);
      if (auto n = routeObj->getInteger("transferLatencyCycles"))
        route.transferLatencyCycles = static_cast<unsigned>(*n);
      if (auto *hopsArr = routeObj->getArray("hops")) {
        for (const auto &hopEntry : *hopsArr) {
          auto *hopPair = hopEntry.getAsArray();
          if (!hopPair || hopPair->size() < 2)
            continue;
          int r = 0, c = 0;
          if (auto n = (*hopPair)[0].getAsInteger())
            r = static_cast<int>(*n);
          if (auto n = (*hopPair)[1].getAsInteger())
            c = static_cast<int>(*n);
          route.hops.push_back({r, c});
        }
      }
      plan.nocPaths.push_back(std::move(route));
    }
  }

  // objectiveValue.
  if (auto *ovObj = obj->getObject("objectiveValue")) {
    if (auto n = ovObj->getNumber("latency"))
      plan.objectiveValue.latency = *n;
    if (auto n = ovObj->getNumber("nocCost"))
      plan.objectiveValue.nocCost = *n;
    if (auto n = ovObj->getNumber("localityBonus"))
      plan.objectiveValue.localityBonus = *n;
  }

  return plan;
}

} // namespace loom
