#ifndef LOOM_SYSTEMCOMPILER_ASSIGNMENTPLAN_H
#define LOOM_SYSTEMCOMPILER_ASSIGNMENTPLAN_H

#include "loom/SystemCompiler/L1CoreAssignment.h"
#include "loom/SystemCompiler/NoCScheduler.h"

#include <map>
#include <string>
#include <vector>

namespace llvm {
namespace json {
class Value;
} // namespace json
} // namespace llvm

namespace loom {

/// Extended assignment plan that consolidates the L1 assignment result
/// with scheduling order and NoC path information.
///
/// The older AssignmentResult type is kept temporarily for backward
/// compatibility but will be removed once all consumers migrate to
/// AssignmentPlan.
struct AssignmentPlan {
  /// Kernel name -> core instance index.
  std::map<std::string, unsigned> kernelToCore;

  /// Per-core assignment details (reuses existing CoreAssignment).
  std::vector<CoreAssignment> coreAssignments;

  /// Topological execution order of kernels.
  std::vector<std::string> schedulingOrder;

  /// NoC routes for each inter-core contract edge.
  std::vector<NoCRoute> nocPaths;

  /// Objective value breakdown.
  struct ObjectiveBreakdown {
    double latency = 0.0;
    double nocCost = 0.0;
    double localityBonus = 0.0;
  };
  ObjectiveBreakdown objectiveValue;

  /// Serialize to JSON.
  llvm::json::Value toJSON() const;

  /// Deserialize from JSON.
  static AssignmentPlan fromJSON(const llvm::json::Value &v);
};

} // namespace loom

#endif // LOOM_SYSTEMCOMPILER_ASSIGNMENTPLAN_H
