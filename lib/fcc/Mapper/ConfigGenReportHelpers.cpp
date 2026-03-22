#include "ConfigGenInternal.h"

#include "fcc/Mapper/OpCompat.h"

#include "mlir/IR/BuiltinAttributes.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <limits>
#include <string>

namespace fcc {
namespace configgen_detail {

std::string escapeJsonString(llvm::StringRef text) {
  std::string escaped;
  escaped.reserve(text.size());
  for (char ch : text) {
    switch (ch) {
    case '\\':
      escaped += "\\\\";
      break;
    case '"':
      escaped += "\\\"";
      break;
    case '\n':
      escaped += "\\n";
      break;
    case '\r':
      escaped += "\\r";
      break;
    case '\t':
      escaped += "\\t";
      break;
    default:
      escaped.push_back(ch);
      break;
    }
  }
  return escaped;
}

std::string summarizeConfigField(const FUConfigField &field) {
  std::string summary;
  llvm::raw_string_ostream os(summary);
  os << static_cast<unsigned>(field.kind) << ":" << field.opIndex << ":"
     << field.templateOpIndex << ":" << field.opName << ":" << field.bitWidth
     << ":" << field.value << ":" << field.sel << ":" << field.discard << ":"
     << field.disconnect;
  return summary;
}

bool getEffectiveFifoBypassed(const Node *hwNode, IdIndex hwId,
                              const MappingState &state) {
  bool bypassable = false;
  bool bypassed = false;
  if (!hwNode)
    return false;
  for (const auto &attr : hwNode->attributes) {
    if (attr.getName() == "bypassable") {
      if (auto boolAttr = mlir::dyn_cast<mlir::BoolAttr>(attr.getValue()))
        bypassable = boolAttr.getValue();
    } else if (attr.getName() == "bypassed") {
      if (auto boolAttr = mlir::dyn_cast<mlir::BoolAttr>(attr.getValue()))
        bypassed = boolAttr.getValue();
    }
  }
  if (!bypassable)
    return false;
  if (hwId < state.hwNodeFifoBypassedOverride.size()) {
    int8_t overrideValue = state.hwNodeFifoBypassedOverride[hwId];
    if (overrideValue == 0)
      return false;
    if (overrideValue > 0)
      return true;
  }
  return bypassed;
}

void writeOptionalUIntJson(llvm::raw_ostream &out, unsigned value) {
  if (value == std::numeric_limits<unsigned>::max()) {
    out << "null";
    return;
  }
  out << value;
}

int64_t computeSelectedUnitTemporalPenalty(const TechMapper::Unit &unit) {
  const auto *preferredCandidate = TechMapper::findPreferredUnitCandidate(unit);
  if (!preferredCandidate)
    return 0;
  return preferredCandidate->temporal ? 64 : 0;
}

int64_t computeFamilyScarcityPenalty(const TechMapper::Plan *techMapPlan,
                                     unsigned familyIndex) {
  if (!techMapPlan)
    return 0;
  llvm::ArrayRef<TechMapper::FamilyTechInfo> familyTechInfos =
      TechMapper::allFamilyTechInfos(*techMapPlan);
  if (familyIndex >= familyTechInfos.size())
    return 0;
  unsigned hwSupportCount = familyTechInfos[familyIndex].hwSupportCount;
  if (hwSupportCount >= 4)
    return 0;
  return static_cast<int64_t>(4 - hwSupportCount) * 48;
}

void writeOptionalUIntText(llvm::raw_ostream &out, unsigned value) {
  if (value == std::numeric_limits<unsigned>::max()) {
    out << "-";
    return;
  }
  out << value;
}

llvm::StringRef lookupSupportClassKey(const TechMapper::Plan *techMapPlan,
                                      unsigned id) {
  if (!techMapPlan)
    return {};
  if (const auto *info = TechMapper::findSupportClass(*techMapPlan, id))
    return info->key;
  return {};
}

llvm::StringRef lookupConfigClassKey(const TechMapper::Plan *techMapPlan,
                                     unsigned id) {
  if (!techMapPlan)
    return {};
  if (const auto *info = TechMapper::findConfigClass(*techMapPlan, id))
    return info->key;
  return {};
}

llvm::StringRef lookupConfigClassReason(const TechMapper::Plan *techMapPlan,
                                        unsigned id) {
  if (!techMapPlan)
    return {};
  if (const auto *info = TechMapper::findConfigClass(*techMapPlan, id))
    return info->reason;
  return {};
}

llvm::StringRef lookupFamilySignature(const TechMapper::Plan *techMapPlan,
                                      unsigned id) {
  if (!techMapPlan)
    return {};
  if (const auto *info = TechMapper::findFamilyTechInfo(*techMapPlan, id))
    return info->signature;
  return {};
}

const TechMapper::CandidateSummaryInfo *
lookupCandidateSummary(const TechMapper::Plan *techMapPlan, unsigned id) {
  if (!techMapPlan)
    return nullptr;
  return TechMapper::findCandidateSummary(*techMapPlan, id);
}

void writeCandidateFeedbackRefJson(llvm::raw_ostream &out,
                                   const TechMapper::Plan *techMapPlan,
                                   unsigned id) {
  const auto *candidate = lookupCandidateSummary(techMapPlan, id);
  out << "{\"id\": " << id;
  if (candidate) {
    out << ", \"family_index\": ";
    writeOptionalUIntJson(out, candidate->familyIndex);
    out << ", \"family_signature\": \""
        << escapeJsonString(candidate->familySignature) << "\"";
    out << ", \"selection_component_id\": ";
    writeOptionalUIntJson(out, candidate->selectionComponentId);
    out << ", \"contracted_node\": ";
    if (candidate->contractedNodeId == INVALID_ID)
      out << "null";
    else
      out << candidate->contractedNodeId;
    out << ", \"support_class_id\": ";
    writeOptionalUIntJson(out, candidate->supportClassId);
    out << ", \"support_class_key\": \""
        << escapeJsonString(candidate->supportClassKey) << "\"";
    out << ", \"config_class_id\": ";
    writeOptionalUIntJson(out, candidate->configClassId);
    out << ", \"config_class_key\": \""
        << escapeJsonString(candidate->configClassKey) << "\"";
    out << ", \"config_class_reason\": \""
        << escapeJsonString(candidate->configClassReason) << "\"";
    out << ", \"status\": \"" << escapeJsonString(candidate->status) << "\"";
    out << ", \"origin_kind\": \""
        << escapeJsonString(TechMapper::originKind(
               candidate->demandOrigin, candidate->legacyFallbackOrigin,
               candidate->mixedOrigin))
        << "\"";
    out << ", \"sw_nodes\": [";
    for (size_t idx = 0; idx < candidate->swNodeIds.size(); ++idx) {
      if (idx != 0)
        out << ", ";
      out << candidate->swNodeIds[idx];
    }
    out << "]";
  }
  out << "}";
}

void writeFamilyFeedbackRefJson(llvm::raw_ostream &out,
                                const TechMapper::Plan *techMapPlan,
                                unsigned id) {
  out << "{\"id\": " << id << ", \"signature\": \""
      << escapeJsonString(lookupFamilySignature(techMapPlan, id)) << "\"}";
}

void writeConfigClassFeedbackRefJson(llvm::raw_ostream &out,
                                     const TechMapper::Plan *techMapPlan,
                                     unsigned id) {
  out << "{\"id\": " << id << ", \"key\": \""
      << escapeJsonString(lookupConfigClassKey(techMapPlan, id)) << "\""
      << ", \"reason\": \""
      << escapeJsonString(lookupConfigClassReason(techMapPlan, id)) << "\"}";
}

void writeCandidatePenaltyJson(llvm::raw_ostream &out,
                               const TechMapper::Plan *techMapPlan,
                               const TechMapper::WeightedIdPenalty &penalty) {
  out << "{\"id\": " << penalty.id << ", \"penalty\": " << penalty.penalty;
  const auto *candidate = lookupCandidateSummary(techMapPlan, penalty.id);
  if (candidate) {
    out << ", \"family_index\": ";
    writeOptionalUIntJson(out, candidate->familyIndex);
    out << ", \"family_signature\": \""
        << escapeJsonString(candidate->familySignature) << "\"";
    out << ", \"selection_component_id\": ";
    writeOptionalUIntJson(out, candidate->selectionComponentId);
    out << ", \"contracted_node\": ";
    if (candidate->contractedNodeId == INVALID_ID)
      out << "null";
    else
      out << candidate->contractedNodeId;
    out << ", \"config_class_id\": ";
    writeOptionalUIntJson(out, candidate->configClassId);
    out << ", \"config_class_key\": \""
        << escapeJsonString(candidate->configClassKey) << "\"";
    out << ", \"status\": \"" << escapeJsonString(candidate->status) << "\"";
  }
  out << "}";
}

void writeFamilyPenaltyJson(llvm::raw_ostream &out,
                            const TechMapper::Plan *techMapPlan,
                            const TechMapper::WeightedIdPenalty &penalty) {
  out << "{\"id\": " << penalty.id << ", \"penalty\": " << penalty.penalty
      << ", \"signature\": \""
      << escapeJsonString(lookupFamilySignature(techMapPlan, penalty.id))
      << "\"}";
}

void writeConfigClassPenaltyJson(llvm::raw_ostream &out,
                                 const TechMapper::Plan *techMapPlan,
                                 const TechMapper::WeightedIdPenalty &penalty) {
  out << "{\"id\": " << penalty.id << ", \"penalty\": " << penalty.penalty
      << ", \"key\": \""
      << escapeJsonString(lookupConfigClassKey(techMapPlan, penalty.id))
      << "\""
      << ", \"reason\": \""
      << escapeJsonString(lookupConfigClassReason(techMapPlan, penalty.id))
      << "\"}";
}

void writeCandidateFeedbackRefText(llvm::raw_ostream &out,
                                   const TechMapper::Plan *techMapPlan,
                                   unsigned id) {
  out << id;
  const auto *candidate = lookupCandidateSummary(techMapPlan, id);
  if (!candidate)
    return;
  out << "[family=";
  writeOptionalUIntText(out, candidate->familyIndex);
  if (!candidate->familySignature.empty())
    out << "/" << candidate->familySignature;
  out << ",component=";
  writeOptionalUIntText(out, candidate->selectionComponentId);
  out << ",cfg=";
  writeOptionalUIntText(out, candidate->configClassId);
  if (!candidate->configClassKey.empty())
    out << "/" << candidate->configClassKey;
  out << ",status=" << candidate->status << "]";
}

void writeFamilyFeedbackRefText(llvm::raw_ostream &out,
                                const TechMapper::Plan *techMapPlan,
                                unsigned id) {
  out << id;
  llvm::StringRef signature = lookupFamilySignature(techMapPlan, id);
  if (!signature.empty())
    out << "[" << signature << "]";
}

void writeConfigClassFeedbackRefText(llvm::raw_ostream &out,
                                     const TechMapper::Plan *techMapPlan,
                                     unsigned id) {
  out << id;
  llvm::StringRef key = lookupConfigClassKey(techMapPlan, id);
  llvm::StringRef reason = lookupConfigClassReason(techMapPlan, id);
  if (!key.empty() || !reason.empty()) {
    out << "[";
    if (!key.empty())
      out << key;
    if (!reason.empty()) {
      if (!key.empty())
        out << "/";
      out << reason;
    }
    out << "]";
  }
}

std::string
sanitizeTechMapDiagnosticsForArtifact(llvm::StringRef diagnostics) {
  if (diagnostics.empty())
    return {};
  llvm::SmallVector<llvm::StringRef, 16> parts;
  diagnostics.split(parts, ", ");
  std::string sanitized;
  llvm::raw_string_ostream os(sanitized);
  bool first = true;
  for (llvm::StringRef part : parts) {
    if (part.starts_with("total_layer2_us=") ||
        part.starts_with("candidate_gen_us=") ||
        part.starts_with("selection_us=")) {
      continue;
    }
    if (!first)
      os << ", ";
    os << part;
    first = false;
  }
  return os.str();
}

} // namespace configgen_detail
} // namespace fcc
