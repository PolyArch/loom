#include "Mapping/Artifact/MappingArtifact.h"

#include "Dataflow/IR/DataflowServiceSchema.h"

#include "llvm/Support/Error.h"

namespace loom::mapping {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "mapping_artifact_invalid: " + message);
}

llvm::Expected<::dataflow::semantics::CanonicalService>
resolveService(const ::dataflow::CanonicalDataflowProgramView &dataflow,
               const TechMemoryActorView &actor,
               ::dataflow::ActorRef terminalActor) {
  if (terminalActor != actor.actor)
    return invalid("memory terminal belongs to another Tech actor");
  auto resolved = dataflow.resolve(actor.actor);
  if (!resolved)
    return resolved.takeError();
  return ::dataflow::semantics::CanonicalService::forActor(resolved->op);
}

} // namespace

llvm::Expected<::loom::fabric::FabricMemoryEngineTemplateEndpointRef>
resolveTechMemoryActorTerminal(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const TechMemoryActorView &actor,
    const ::dataflow::ActorTokenOperandRef &terminal) {
  auto service = resolveService(dataflow, actor, terminal.actor);
  if (!service)
    return service.takeError();
  auto resolved = dataflow.resolve(actor.actor);
  if (!resolved)
    return resolved.takeError();
  for (unsigned ordinal = 0; ordinal < service->arguments().size(); ++ordinal) {
    auto value = service->argumentValue(resolved->op, ordinal);
    if (!value)
      return value.takeError();
    if ((*value)->getOperandNumber() != terminal.ordinal)
      continue;
    if (ordinal >= actor.operandPorts.size())
      return invalid("memory actor operand correspondence is incomplete");
    return actor.operandPorts[ordinal];
  }
  return invalid("memory terminal names a non-service actor operand");
}

llvm::Expected<::loom::fabric::FabricMemoryEngineTemplateEndpointRef>
resolveTechMemoryActorTerminal(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const TechMemoryActorView &actor,
    const ::dataflow::ActorTokenResultRef &terminal) {
  auto service = resolveService(dataflow, actor, terminal.actor);
  if (!service)
    return service.takeError();
  auto resolved = dataflow.resolve(actor.actor);
  if (!resolved)
    return resolved.takeError();
  for (unsigned ordinal = 0; ordinal < service->results().size(); ++ordinal) {
    auto value = service->resultValue(resolved->op, ordinal);
    if (!value)
      return value.takeError();
    if (value->getResultNumber() != terminal.ordinal)
      continue;
    if (ordinal >= actor.resultPorts.size())
      return invalid("memory actor result correspondence is incomplete");
    return actor.resultPorts[ordinal];
  }
  return invalid("memory terminal names a non-service actor result");
}

} // namespace loom::mapping
