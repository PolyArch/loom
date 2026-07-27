#include "Frontend/Compilation/StaticMemoryBinding.h"

#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Dataflow/IR/DataflowOps.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <system_error>

namespace loom::frontend {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      llvm::Twine("static_memory_binding_invalid: ") + message);
}

std::optional<std::uint64_t>
globalOrdinal(const StaticGlobalMemoryCatalog &catalog,
              llvm::StringRef symbol) {
  const StaticGlobalMemory *global = catalog.lookup(symbol);
  if (!global)
    return std::nullopt;
  return static_cast<std::uint64_t>(global - catalog.globals.data());
}

} // namespace

llvm::Expected<std::vector<RootedLogicalMemorySource>>
deriveRootedLogicalMemorySources(
    const StaticGlobalMemoryCatalog &catalog,
    const dataflow::CanonicalDataflowProgramView &program,
    dataflow::RootedGraphLaunchRef launch) {
  auto graph = program.resolve(launch);
  if (!graph)
    return graph.takeError();
  auto rootLaunch = program.resolve(launch.rootThreadLaunch);
  if (!rootLaunch)
    return rootLaunch.takeError();
  auto launchOp = llvm::dyn_cast<dataflow::ThreadLaunchOp>(rootLaunch->op);
  auto threadOp = llvm::dyn_cast<dataflow::ThreadOp>(rootLaunch->callee);
  if (!launchOp || !threadOp)
    return invalid("root launch view does not resolve its thread relation");

  std::vector<RootedLogicalMemorySource> sources;
  for (const dataflow::CanonicalLogicalMemoryRootView &root :
       program.logicalMemoryRoots()) {
    if (root.op != threadOp.getOperation() || !root.formalArgIndex)
      continue;
    const unsigned argument = *root.formalArgIndex;
    if (argument >= launchOp.getBodyOperands().size())
      return invalid("logical memory formal exceeds launch body operands");

    RootedLogicalMemorySource source{root.ref, std::nullopt};
    mlir::Value operand = launchOp.getBodyOperands()[argument];
    if (auto address = operand.getDefiningOp<mlir::LLVM::AddressOfOp>()) {
      source.globalOrdinal = globalOrdinal(catalog, address.getGlobalName());
      if (!source.globalOrdinal)
        return invalid(llvm::Twine("launch references global '") +
                       address.getGlobalName() +
                       "' absent from the linked LLVM catalog");
    }
    sources.push_back(source);
  }

  llvm::sort(sources, [](const RootedLogicalMemorySource &lhs,
                         const RootedLogicalMemorySource &rhs) {
    return lhs.root.entity.value() < rhs.root.entity.value();
  });
  return sources;
}

} // namespace loom::frontend
