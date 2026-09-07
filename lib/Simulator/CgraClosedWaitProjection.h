#ifndef LOOM_LIB_SIMULATOR_CGRACLOSEDWAITPROJECTION_H
#define LOOM_LIB_SIMULATOR_CGRACLOSEDWAITPROJECTION_H

#include "Simulator/CGRASimulator.h"

namespace loom::sim::detail {

struct PreparedGraphExecution;
struct SimulatorState;
class CgraGraphActivationRuntime;

/// Projects a quiescent mapped execution into the existing typed wait
/// evidence, retaining all physical and semantic owner occurrences. This
/// projection does not advance the activation or decide its lifecycle.
llvm::Expected<CgraClosedWaitSetDiagnostic> projectCgraClosedWaitSet(
    const PreparedGraphExecution &execution, const SimulatorState &dynamicState,
    const CgraGraphActivationRuntime &runtime,
    CgraExecutionOwnerReferences ownerReferences, bool graphRetirementVisible);

} // namespace loom::sim::detail

#endif // LOOM_LIB_SIMULATOR_CGRACLOSEDWAITPROJECTION_H
