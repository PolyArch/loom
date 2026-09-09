#include "DataflowRewriteInternal.h"

#include "Dataflow/IR/DataflowGraphValidation.h"
#include "Dataflow/IR/DataflowOps.h"

#include "mlir/IR/Builders.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

namespace dataflow::detail {
namespace {

struct CompletionCollectors {
  llvm::SmallVector<CarryOp> carries;
  llvm::SmallVector<DemuxOp> exits;
};

CompletionCollectors collectCompletions(StreamOp stream) {
  CompletionCollectors result;
  for (mlir::Operation *user : stream.getPhase().getUsers()) {
    auto carry = llvm::dyn_cast<CarryOp>(user);
    if (!carry || carry.getCond() != stream.getPhase() ||
        !llvm::isa<mlir::NoneType>(carry.getOutput().getType()) ||
        carry.getOutput().use_empty())
      continue;
    llvm::SmallVector<DemuxOp> exits;
    bool completionOnly = llvm::all_of(
        carry.getOutput().getUsers(), [&](mlir::Operation *consumer) {
          auto demux = llvm::dyn_cast<DemuxOp>(consumer);
          if (!demux || demux.getInput() != carry.getOutput() ||
              demux.getSel() != stream.getPhase() ||
              demux.getOutputs().size() != 2 ||
              !demux.getOutputs()[1].use_empty())
            return false;
          exits.push_back(demux);
          return true;
        });
    if (completionOnly) {
      result.carries.push_back(carry);
      result.exits.append(exits);
    }
  }
  return result;
}

StreamOp splitCompletionPhase(StreamOp stream,
                              const CompletionCollectors &collectors) {
  mlir::OpBuilder builder(stream);
  auto completion = StreamOp::create(
      builder, stream.getLoc(), stream.getIv().getType(),
      stream.getPhase().getType(), stream.getInit(), stream.getLimit(),
      stream.getStep(), stream.getStepKind(), stream.getPredicate());
  for (CarryOp carry : collectors.carries)
    carry.getCondMutable().assign(completion.getPhase());
  for (DemuxOp exit : collectors.exits)
    exit.getSelMutable().assign(completion.getPhase());
  return completion;
}

// Check both physical instances against the existing causal-close owner on a
// graph-local clone. The proof does not infer retirement from equal counts.
bool hasSplitCloseWitnesses(StreamOp stream) {
  auto graph = stream->getParentOfType<GraphOp>();
  mlir::IRMapping mapping;
  mlir::OwningOpRef<GraphOp> candidate(
      llvm::cast<GraphOp>(graph->clone(mapping)));
  auto source = llvm::cast<StreamOp>(mapping.lookup(stream.getOperation()));
  auto collectors = collectCompletions(source);
  if (collectors.carries.empty())
    return false;
  auto completion = splitCompletionPhase(source, collectors);
  auto ret =
      llvm::cast<GraphReturnOp>(candidate->getBody().front().getTerminator());
  return retirementCoversClose(source.getPhase(), ret.getComplete()) &&
         retirementCoversClose(completion.getPhase(), ret.getComplete());
}

} // namespace

llvm::Expected<std::vector<DataflowRewriteDecision>>
enumerateStreamCompletionPhaseSplitDecisions(
    const CanonicalDataflowArtifact &parent) {
  std::vector<DataflowRewriteDecision> decisions;
  for (const CanonicalActorView &actor : parent.view().actors()) {
    auto stream = llvm::dyn_cast<StreamOp>(actor.op);
    if (!stream || collectCompletions(stream).carries.empty())
      continue;
    if (hasSplitCloseWitnesses(stream))
      decisions.emplace_back(
          StreamCompletionPhaseSplitRewrite{actor.ref.entity});
  }
  llvm::sort(decisions, dataflowRewriteDecisionLess);
  return decisions;
}

llvm::Expected<std::optional<MaterializedDataflowRewriteProjection>>
materializeStreamCompletionPhaseSplitProjection(
    const CanonicalDataflowArtifact &parent,
    const StreamCompletionPhaseSplitRewrite &decision,
    llvm::ArrayRef<StaticGraphLaunchRef> trackedStaticGraphLaunches,
    llvm::ArrayRef<mlir::Value> trackedValues) {
  auto actor =
      parent.view().resolve(ActorRef{parent.identity(), decision.stream});
  if (!actor)
    return actor.takeError();
  auto stream = llvm::dyn_cast<StreamOp>(actor->op);
  if (!stream || !hasSplitCloseWitnesses(stream))
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "dataflow_stream_rewrite_invalid: source lacks separable completion "
        "collectors with independent close witnesses");
  mlir::IRMapping mapping;
  mlir::OwningOpRef<mlir::ModuleOp> candidate(
      llvm::cast<mlir::ModuleOp>(parent.module()->clone(mapping)));
  auto source = llvm::cast<StreamOp>(mapping.lookup(stream.getOperation()));
  splitCompletionPhase(source, collectCompletions(source));
  return finalizeDataflowRewriteCandidate(
      parent, *candidate, mapping, trackedStaticGraphLaunches, trackedValues);
}

} // namespace dataflow::detail
