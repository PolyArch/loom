#include "Dataflow/IR/DataflowCanonicalArtifact.h"

#include "Dataflow/IR/DataflowOps.h"

#include "mlir/IR/BuiltinTypes.h"

#include "llvm/Support/Error.h"

#include <cstdint>
#include <limits>
#include <vector>

namespace dataflow {

llvm::Expected<CanonicalRootThreadLogicalDomainView>
CanonicalDataflowProgramView::projectRootThreadLogicalDomain(
    RootThreadLaunchRef ref) const {
  auto resolved = resolve(ref);
  if (!resolved)
    return resolved.takeError();
  auto launch = llvm::dyn_cast<ThreadLaunchOp>(resolved->op);
  auto thread = llvm::dyn_cast<ThreadOp>(resolved->callee);
  if (!launch || !thread || thread.isExternal())
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "dataflow_logical_domain_invalid: root launch does not resolve a "
        "body-owning thread");

  const std::size_t inputCount = thread.getFunctionType().getNumInputs();
  const std::size_t bodyArgumentCount =
      thread.getBody().front().getNumArguments();
  if (bodyArgumentCount < inputCount + 1)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "dataflow_logical_domain_invalid: thread body lacks its control slot");
  const std::size_t rank = bodyArgumentCount - inputCount - 1;
  if (rank > std::numeric_limits<std::uint32_t>::max())
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "dataflow_logical_domain_invalid: coordinate rank exceeds u32");
  if (launch.getGridUpperBounds().size() != rank)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "dataflow_logical_domain_invalid: launch extent count differs from "
        "thread coordinate rank");

  std::vector<mlir::Value> parameters;
  parameters.reserve(launch.getGridUpperBounds().size() +
                     launch.getBodyOperands().size());
  parameters.insert(parameters.end(), launch.getGridUpperBounds().begin(),
                    launch.getGridUpperBounds().end());

  const std::optional<std::uint64_t> workItemOrdinal =
      thread.getDomain().getWorkItemArgOrdinal();
  for (auto item : llvm::enumerate(launch.getBodyOperands())) {
    if (thread.getDomain().getKind() == ThreadDomainKind::DynamicWork &&
        workItemOrdinal && item.index() == *workItemOrdinal)
      continue;
    mlir::Type type = item.value().getType();
    auto integer = llvm::dyn_cast<mlir::IntegerType>(type);
    if (llvm::isa<mlir::IndexType>(type) || (integer && integer.isSignless()))
      parameters.push_back(item.value());
  }

  return CanonicalRootThreadLogicalDomainView{ref, thread.getDomain().getKind(),
                                              static_cast<std::uint32_t>(rank),
                                              std::move(parameters)};
}

llvm::Expected<std::optional<CanonicalRootThreadLogicalDomainView>>
CanonicalDataflowProgramView::projectWholeRootedGraphLogicalDomain(
    RootedGraphLaunchRef ref) const {
  auto rooted = resolve(ref);
  if (!rooted)
    return rooted.takeError();
  auto root = resolve(ref.rootThreadLaunch);
  if (!root)
    return root.takeError();
  auto graphLaunch = resolve(ref.staticGraphLaunch);
  if (!graphLaunch)
    return graphLaunch.takeError();

  auto thread = llvm::dyn_cast<ThreadOp>(root->callee);
  auto launch = llvm::dyn_cast<GraphLaunchOp>(graphLaunch->op);
  if (!thread || thread.isExternal() || !launch)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "dataflow_logical_domain_invalid: rooted graph launch does not "
        "resolve inside a body-owning thread");

  if (launch->getParentOp() != thread.getOperation() ||
      launch->getBlock() != &thread.getBody().front())
    return std::optional<CanonicalRootThreadLogicalDomainView>{};

  auto domain = projectRootThreadLogicalDomain(ref.rootThreadLaunch);
  if (!domain)
    return domain.takeError();
  return std::optional<CanonicalRootThreadLogicalDomainView>(
      std::move(*domain));
}

} // namespace dataflow
