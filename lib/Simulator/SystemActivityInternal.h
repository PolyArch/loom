#ifndef LOOM_LIB_SIMULATOR_SYSTEMACTIVITYINTERNAL_H
#define LOOM_LIB_SIMULATOR_SYSTEMACTIVITYINTERNAL_H

#include "llvm/Support/Error.h"
namespace loom::sim {
struct SystemSimulationExecution;
namespace detail {
struct SystemExecutionContext;
llvm::Error validateSystemMemoryActivity(const SystemSimulationExecution &execution,
                                        const SystemExecutionContext &context);
}
}
#endif
