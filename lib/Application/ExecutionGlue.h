#ifndef LOOM_LIB_APPLICATION_EXECUTIONGLUE_H
#define LOOM_LIB_APPLICATION_EXECUTIONGLUE_H

#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Runtime/SpatialInvocationWire.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace llvm {
class Module;
}

namespace loom::application::detail {

inline constexpr llvm::StringLiteral applicationHostEntrySymbol{
    "__loom_host_entry"};

struct ApplicationSpatialInvocationPlan final {
  dataflow::RootThreadLaunchRef root;
  dataflow::RootedGraphLaunchRef graph;
  std::string sourceCallableSymbol;
  std::uint64_t dispatchTargetOrdinal = 0;
  std::vector<std::uint32_t> valueBitCounts;
  std::vector<std::uint32_t> resultBitCounts;
  runtime::SpatialInvocationWireLayout wireLayout;
};

llvm::Expected<ApplicationSpatialInvocationPlan>
deriveApplicationSpatialInvocationPlan(
    const dataflow::CanonicalDataflowProgramView &dataflow,
    llvm::StringRef entrySymbol);

llvm::Expected<std::unique_ptr<llvm::Module>>
materializeHostDispatchModule(const llvm::Module &finalLinkedModule,
                              llvm::StringRef applicationEntry,
                              const ApplicationSpatialInvocationPlan &plan);

llvm::Expected<std::unique_ptr<llvm::Module>>
materializeInstructionDispatchModule(const llvm::Module &finalLinkedModule,
                                     std::uint64_t entryCount);

} // namespace loom::application::detail

#endif // LOOM_LIB_APPLICATION_EXECUTIONGLUE_H
