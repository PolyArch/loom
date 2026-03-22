#include "loom/Simulator/SimTypes.h"

namespace loom {
namespace sim {

const char *eventKindName(EventKind kind) {
  switch (kind) {
  case EventKind::NodeFire:
    return "node_fire";
  case EventKind::NodeStallIn:
    return "node_stall_in";
  case EventKind::NodeStallOut:
    return "node_stall_out";
  case EventKind::RouteUse:
    return "route_use";
  case EventKind::ConfigWrite:
    return "config_write";
  case EventKind::InvocationStart:
    return "invocation_start";
  case EventKind::InvocationDone:
    return "invocation_done";
  case EventKind::DeviceError:
    return "device_error";
  }
  return "unknown";
}

const char *simPhaseName(SimPhase phase) {
  switch (phase) {
  case SimPhase::Evaluate:
    return "evaluate";
  case SimPhase::Commit:
    return "commit";
  }
  return "unknown";
}

const char *boundaryReasonName(BoundaryReason reason) {
  switch (reason) {
  case BoundaryReason::NeedMemIssue:
    return "need_mem_issue";
  case BoundaryReason::WaitMemResp:
    return "wait_mem_resp";
  case BoundaryReason::InvocationDone:
    return "invocation_done";
  case BoundaryReason::Deadlock:
    return "deadlock";
  case BoundaryReason::BudgetHit:
    return "budget_hit";
  case BoundaryReason::None:
    return "none";
  }
  return "unknown";
}

} // namespace sim
} // namespace loom
