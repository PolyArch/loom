#ifndef LOOM_SYSTEMCOMPILER_INFEASIBILITYCUT_H
#define LOOM_SYSTEMCOMPILER_INFEASIBILITYCUT_H

#include <cstdint>
#include <string>
#include <variant>
#include <vector>

namespace llvm {
namespace json {
class Value;
} // namespace json
} // namespace llvm

namespace loom {

/// Reason for infeasibility in the L2 compiler.
enum class CutReason {
  INSUFFICIENT_FU,    // Not enough FUs of required type
  ROUTING_CONGESTION, // Switch network saturated
  SPM_OVERFLOW,       // SPM capacity exceeded
  II_UNACHIEVABLE,    // Cannot achieve target II
  TYPE_MISMATCH,      // Core type lacks required operation
};

/// Quantitative evidence for an infeasibility cut.
struct FUShortage {
  std::string fuType;
  unsigned needed;
  unsigned available;
};

struct CongestionInfo {
  double utilizationPct;
};

struct SPMInfo {
  uint64_t neededBytes;
  uint64_t availableBytes;
};

struct IIInfo {
  unsigned minII;
  unsigned targetII;
};

/// Constraint suggesting a kernel should be excluded from a core type.
struct ExclusionConstraint {
  std::string kernel;
  std::string coreType;
};

/// Constraint suggesting a core type needs more of a specific resource.
struct CapacityConstraint {
  std::string coreType;
  std::string resourceName;
  unsigned minRequired = 0;
};

/// An infeasibility cut reports why a kernel cannot be mapped to a core.
struct InfeasibilityCut {
  std::string kernelName;
  std::string coreType;
  CutReason reason;
  std::variant<FUShortage, CongestionInfo, SPMInfo, IIInfo> evidence;

  /// Suggested constraints for the L1 solver to add in subsequent iterations.
  std::vector<std::variant<ExclusionConstraint, CapacityConstraint>>
      suggestedConstraints;
};

// Enum <-> string conversion
const char *cutReasonToString(CutReason r);
CutReason cutReasonFromString(const std::string &s);

// JSON serialization
llvm::json::Value infeasibilityCutToJSON(const InfeasibilityCut &cut);
InfeasibilityCut infeasibilityCutFromJSON(const llvm::json::Value &v);

} // namespace loom

#endif
