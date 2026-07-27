#ifndef LOOM_FRONTEND_COMPILATION_STATICMEMORYBINDING_H
#define LOOM_FRONTEND_COMPILATION_STATICMEMORYBINDING_H

#include "Dataflow/IR/DataflowCanonicalEntity.h"
#include "Dataflow/IR/DataflowStructuralRefs.h"
#include "Frontend/Compilation/StaticGlobalMemory.h"

#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace dataflow {
class CanonicalDataflowProgramView;
} // namespace dataflow

namespace loom::frontend {

/// The launch-local source of one imported logical memory root. A missing
/// global ordinal means the launch supplies a dynamic runtime capability.
/// A present ordinal indexes the exact invocation-local StaticGlobalMemory
/// catalog; the referenced record itself states Image versus ExternalRuntime.
struct RootedLogicalMemorySource {
  dataflow::LogicalMemoryRootRef root;
  std::optional<std::uint64_t> globalOrdinal;
};

/// Derives the total imported-memory source relation for one rooted graph
/// launch from the canonical Dataflow launch binding. This never guesses from
/// a graph operand type or private symbol spelling in isolation.
llvm::Expected<std::vector<RootedLogicalMemorySource>>
deriveRootedLogicalMemorySources(
    const StaticGlobalMemoryCatalog &catalog,
    const dataflow::CanonicalDataflowProgramView &program,
    dataflow::RootedGraphLaunchRef launch);

} // namespace loom::frontend

#endif // LOOM_FRONTEND_COMPILATION_STATICMEMORYBINDING_H
