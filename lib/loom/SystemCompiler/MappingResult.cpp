#include "loom/SystemCompiler/MappingResult.h"
#include "llvm/Support/JSON.h"

#include <algorithm>
#include <string>

using llvm::json::Array;
using llvm::json::Object;
using llvm::json::Value;

namespace loom {

/// Base64 encoding for config blob serialization.
static const char kBase64Table[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

static std::string base64Encode(const std::vector<uint8_t> &data) {
  std::string result;
  result.reserve((data.size() + 2) / 3 * 4);
  for (size_t i = 0; i < data.size(); i += 3) {
    uint32_t n = static_cast<uint32_t>(data[i]) << 16;
    if (i + 1 < data.size())
      n |= static_cast<uint32_t>(data[i + 1]) << 8;
    if (i + 2 < data.size())
      n |= static_cast<uint32_t>(data[i + 2]);

    result.push_back(kBase64Table[(n >> 18) & 0x3F]);
    result.push_back(kBase64Table[(n >> 12) & 0x3F]);
    result.push_back((i + 1 < data.size()) ? kBase64Table[(n >> 6) & 0x3F]
                                           : '=');
    result.push_back((i + 2 < data.size()) ? kBase64Table[n & 0x3F] : '=');
  }
  return result;
}

static uint8_t base64DecodeChar(char c) {
  if (c >= 'A' && c <= 'Z')
    return static_cast<uint8_t>(c - 'A');
  if (c >= 'a' && c <= 'z')
    return static_cast<uint8_t>(c - 'a' + 26);
  if (c >= '0' && c <= '9')
    return static_cast<uint8_t>(c - '0' + 52);
  if (c == '+')
    return 62;
  if (c == '/')
    return 63;
  return 0;
}

static std::vector<uint8_t> base64Decode(const std::string &encoded) {
  std::vector<uint8_t> result;
  if (encoded.empty())
    return result;
  result.reserve(encoded.size() * 3 / 4);
  for (size_t i = 0; i + 3 < encoded.size(); i += 4) {
    uint32_t n = (static_cast<uint32_t>(base64DecodeChar(encoded[i])) << 18) |
                 (static_cast<uint32_t>(base64DecodeChar(encoded[i + 1])) << 12);
    if (encoded[i + 2] != '=')
      n |= static_cast<uint32_t>(base64DecodeChar(encoded[i + 2])) << 6;
    if (encoded[i + 3] != '=')
      n |= static_cast<uint32_t>(base64DecodeChar(encoded[i + 3]));

    result.push_back(static_cast<uint8_t>((n >> 16) & 0xFF));
    if (encoded[i + 2] != '=')
      result.push_back(static_cast<uint8_t>((n >> 8) & 0xFF));
    if (encoded[i + 3] != '=')
      result.push_back(static_cast<uint8_t>(n & 0xFF));
  }
  return result;
}

Value MappingResult::toJSON() const {
  Object obj;
  obj["success"] = success;

  // Resource usage.
  Object resObj;
  resObj["peUtilization"] = resourceUsage.peUtilization;
  resObj["fuUtilization"] = resourceUsage.fuUtilization;
  resObj["spmBytesUsed"] = static_cast<int64_t>(resourceUsage.spmBytesUsed);
  obj["resourceUsage"] = std::move(resObj);

  // Cycle estimate.
  Object cycObj;
  cycObj["achievedII"] = static_cast<int64_t>(cycleEstimate.achievedII);
  cycObj["totalExecutionCycles"] =
      static_cast<int64_t>(cycleEstimate.totalExecutionCycles);
  cycObj["tripCount"] = static_cast<int64_t>(cycleEstimate.tripCount);
  obj["cycleEstimate"] = std::move(cycObj);

  // Routing congestion.
  Object routObj;
  routObj["maxSwitchUtilization"] = routingCongestion.maxSwitchUtilization;
  routObj["unroutedEdgeCount"] =
      static_cast<int64_t>(routingCongestion.unroutedEdgeCount);
  obj["routingCongestion"] = std::move(routObj);

  // Per-kernel results.
  Array perKernelArr;
  for (const auto &km : perKernelResults) {
    perKernelArr.push_back(kernelMetricsToJSON(km));
  }
  obj["perKernelResults"] = std::move(perKernelArr);

  // Config blob (base64 encoded).
  if (configBlob.has_value()) {
    obj["configBlob"] = base64Encode(configBlob.value());
  }

  return Value(std::move(obj));
}

MappingResult MappingResult::fromJSON(const Value &v) {
  MappingResult result;
  auto *obj = v.getAsObject();
  if (!obj)
    return result;

  if (auto b = obj->getBoolean("success"))
    result.success = *b;

  // Resource usage.
  if (auto *resObj = obj->getObject("resourceUsage")) {
    if (auto n = resObj->getNumber("peUtilization"))
      result.resourceUsage.peUtilization = *n;
    if (auto n = resObj->getNumber("fuUtilization"))
      result.resourceUsage.fuUtilization = *n;
    if (auto n = resObj->getInteger("spmBytesUsed"))
      result.resourceUsage.spmBytesUsed = static_cast<uint64_t>(*n);
  }

  // Cycle estimate.
  if (auto *cycObj = obj->getObject("cycleEstimate")) {
    if (auto n = cycObj->getInteger("achievedII"))
      result.cycleEstimate.achievedII = static_cast<unsigned>(*n);
    if (auto n = cycObj->getInteger("totalExecutionCycles"))
      result.cycleEstimate.totalExecutionCycles = static_cast<uint64_t>(*n);
    if (auto n = cycObj->getInteger("tripCount"))
      result.cycleEstimate.tripCount = static_cast<unsigned>(*n);
  }

  // Routing congestion.
  if (auto *routObj = obj->getObject("routingCongestion")) {
    if (auto n = routObj->getNumber("maxSwitchUtilization"))
      result.routingCongestion.maxSwitchUtilization = *n;
    if (auto n = routObj->getInteger("unroutedEdgeCount"))
      result.routingCongestion.unroutedEdgeCount = static_cast<unsigned>(*n);
  }

  // Per-kernel results.
  if (auto *arr = obj->getArray("perKernelResults")) {
    for (const auto &entry : *arr) {
      result.perKernelResults.push_back(kernelMetricsFromJSON(entry));
    }
  }

  // Config blob.
  if (auto s = obj->getString("configBlob")) {
    result.configBlob = base64Decode(s->str());
  }

  return result;
}

} // namespace loom
