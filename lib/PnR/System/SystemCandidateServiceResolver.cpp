#include "SystemCandidateServiceResolver.h"

#include "Fabric/Identity/FabricRefBytes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <optional>
#include <utility>
#include <vector>

using namespace loom;
using namespace loom::pnr;

namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "system_candidate_invalid: " + message);
}

bool sameSubject(const SystemServiceTargetSubject &left,
                 const SystemServiceTargetSubject &right) {
  if (left.index() != right.index())
    return false;
  if (const auto *member = std::get_if<SystemServiceMemberTargetSubject>(&left))
    return member->member ==
           std::get<SystemServiceMemberTargetSubject>(right).member;
  return std::get<SystemMemoryExposureTargetSubject>(left).exposure ==
         std::get<SystemMemoryExposureTargetSubject>(right).exposure;
}

struct SelectedServiceBinding final {
  const FrozenSystemServiceContext *context = nullptr;
  const FrozenSystemMemoryServiceBinding *binding = nullptr;
};

llvm::Expected<SelectedServiceBinding>
resolveBinding(const FrozenSystemPnrProblem &problem, PnrIndex contextOrdinal,
               const SystemServiceTargetSubject &subject,
               llvm::ArrayRef<PnrIndex> threadChoices,
               llvm::ArrayRef<PnrIndex> graphChoices) {
  if (contextOrdinal >= problem.serviceContexts().size())
    return invalid("service context is out of range");
  const auto &context = problem.serviceContexts()[contextOrdinal];
  if (context.service >= problem.serviceDomains().size() ||
      context.threadDecision >= problem.threadDecisions().size() ||
      context.graphDecision >= problem.graphDecisions().size() ||
      context.threadDecision >= threadChoices.size() ||
      context.graphDecision >= graphChoices.size())
    return invalid("service context has an invalid execution dependency");
  const auto threadDomain =
      problem.threadChoiceCatalogOrdinals(context.threadDecision);
  const auto graphDomain =
      problem.graphChoiceCatalogOrdinals(context.graphDecision);
  if (threadChoices[context.threadDecision] >= threadDomain.size() ||
      graphChoices[context.graphDecision] >= graphDomain.size())
    return invalid("service context choice is outside its H domain");
  const auto core =
      problem.accCores()[threadDomain[threadChoices[context.threadDecision]]];
  const auto &mapping =
      problem
          .spatialMappings()[graphDomain[graphChoices[context.graphDecision]]];
  const auto &service = problem.serviceDomains()[context.service];

  const FrozenSystemMemoryServiceBinding *selected = nullptr;
  for (const FrozenSystemMemoryServiceBinding &binding :
       problem.memoryServiceBindings()) {
    if (binding.obligation != service.key ||
        !sameSubject(binding.subject, subject) ||
        binding.spatialMapping != mapping || binding.accCore != core)
      continue;
    if (selected &&
        (selected->systemEndpoint != binding.systemEndpoint ||
         selected->occurrenceEndpoint != binding.occurrenceEndpoint))
      return invalid("selected execution resolves more than one memory "
                     "attachment pair");
    selected = &binding;
  }
  if (!selected)
    return invalid("selected execution has no bound memory service endpoint");
  return SelectedServiceBinding{&context, selected};
}

template <typename Ref>
std::vector<Ref> intersectDomains(const std::vector<Ref> &left,
                                  const std::vector<Ref> &right) {
  std::vector<Ref> result;
  result.reserve(std::min(left.size(), right.size()));
  for (const Ref &value : left)
    if (llvm::is_contained(right, value))
      result.push_back(value);
  return result;
}

llvm::Expected<const SystemSearchServiceTargetCompatibility *>
findTargetRow(const SystemSearchServiceDomain &service,
              const SystemServiceTargetSubject &subject,
              ::loom::fabric::SystemServiceEndpointRef endpoint) {
  const SystemSearchServiceTargetCompatibility *result = nullptr;
  for (const auto &row : service.targetCompatibility) {
    if (!sameSubject(row.subject, subject) || row.boundEndpoint != endpoint)
      continue;
    if (result)
      return invalid("H repeats one exact service target row");
    result = &row;
  }
  if (!result)
    return invalid("H is missing the matching service target row");
  return result;
}

llvm::Expected<const SystemSearchTransferTerminalCompatibility *>
findTerminalRow(const SystemSearchServiceDomain &service,
                const ::loom::mapping::SystemTransferTerminalKey &terminal,
                const FrozenSystemMemoryServiceBinding &binding) {
  const ::loom::fabric::FabricMemoryEndpointRef systemEndpoint{
      ::loom::fabric::FabricMemoryEndpointOwnerRef::of(binding.systemEndpoint),
      0};
  const SystemSearchTransferTerminalCompatibility *result = nullptr;
  for (const auto &row : service.transferTerminalCompatibility) {
    if (!(row.terminal == terminal))
      continue;
    const auto *bound =
        std::get_if<SystemMemoryOrFenceTerminalEndpoint>(&row.boundEndpoint);
    if (!bound || (bound->endpoint != binding.occurrenceEndpoint &&
                   bound->endpoint != systemEndpoint))
      continue;
    if (result)
      return invalid("H has ambiguous matching terminal rows");
    result = &row;
  }
  if (!result)
    return invalid("H is missing the matching service terminal row");
  return result;
}

} // namespace

llvm::Expected<SystemServiceTargetDomain>
loom::pnr::detail::resolveSystemServiceTargetDomain(
    const FrozenSystemPnrProblem &problem, PnrIndex contextOrdinal,
    llvm::ArrayRef<PnrIndex> threadChoices,
    llvm::ArrayRef<PnrIndex> graphChoices) {
  if (contextOrdinal >= problem.serviceContexts().size())
    return invalid("service context is out of range");
  const auto &context = problem.serviceContexts()[contextOrdinal];
  if (context.subjects.empty())
    return invalid("service context has no target subjects");
  if (context.service >= problem.serviceDomains().size())
    return invalid("service context has no H service domain");
  const auto &service = problem.serviceDomains()[context.service];

  std::optional<SystemServiceTargetDomain> intersection;
  for (const SystemServiceTargetSubject &subject : context.subjects) {
    auto selected = resolveBinding(problem, contextOrdinal, subject,
                                   threadChoices, graphChoices);
    if (!selected)
      return selected.takeError();
    auto row =
        findTargetRow(service, subject, selected->binding->systemEndpoint);
    if (!row)
      return row.takeError();
    if (!intersection) {
      intersection = (*row)->compatibleTargets;
      continue;
    }
    if (intersection->index() != (*row)->compatibleTargets.index())
      return invalid("one service context mixes target-domain kinds");
    if (auto *regions = std::get_if<
            std::vector<::loom::fabric::FabricMemoryServiceRegionRef>>(
            &*intersection)) {
      *regions = intersectDomains(
          *regions,
          std::get<std::vector<::loom::fabric::FabricMemoryServiceRegionRef>>(
              (*row)->compatibleTargets));
    } else {
      auto &domains =
          std::get<std::vector<::loom::fabric::MemoryConsistencyDomainRef>>(
              *intersection);
      domains = intersectDomains(
          domains,
          std::get<std::vector<::loom::fabric::MemoryConsistencyDomainRef>>(
              (*row)->compatibleTargets));
    }
  }
  if (!intersection)
    return invalid("service context did not resolve a target domain");
  const bool empty = std::visit(
      [](const auto &values) { return values.empty(); }, *intersection);
  if (empty)
    return invalid("matching service target rows have an empty intersection");
  return std::move(*intersection);
}

llvm::Expected<std::vector<PnrIndex>>
loom::pnr::detail::resolveSystemServiceTerminalDomain(
    const FrozenSystemPnrProblem &problem, PnrIndex legOrdinal,
    PnrIndex terminalOrdinal, llvm::ArrayRef<PnrIndex> threadChoices,
    llvm::ArrayRef<PnrIndex> graphChoices) {
  if (legOrdinal >= problem.serviceLegs().size() ||
      terminalOrdinal >= problem.serviceTerminals().size())
    return invalid("service terminal lookup is out of range");
  const auto &leg = problem.serviceLegs()[legOrdinal];
  if (leg.serviceContext == getInvalidPnrIndex())
    return std::vector<PnrIndex>(
        problem.serviceTerminalEndpointChoices(terminalOrdinal).begin(),
        problem.serviceTerminalEndpointChoices(terminalOrdinal).end());

  const SystemServiceTargetSubject subject{
      SystemServiceMemberTargetSubject{leg.key.member}};
  auto selected = resolveBinding(problem, leg.serviceContext, subject,
                                 threadChoices, graphChoices);
  if (!selected)
    return selected.takeError();
  const auto &service = problem.serviceDomains()[selected->context->service];
  const auto &terminal = problem.serviceTerminals()[terminalOrdinal].key;
  auto row = findTerminalRow(service, terminal, *selected->binding);
  if (!row)
    return row.takeError();

  std::vector<PnrIndex> result;
  for (const auto endpoint : (*row)->compatibleTransportEndpoints) {
    const auto found = llvm::find_if(
        problem.routingTopology().endpoints(),
        [&](const auto &candidate) { return candidate.reference == endpoint; });
    if (found == problem.routingTopology().endpoints().end())
      return invalid("matching H terminal row names an endpoint outside F");
    result.push_back(static_cast<PnrIndex>(
        found - problem.routingTopology().endpoints().begin()));
  }
  llvm::sort(result);
  if (std::adjacent_find(result.begin(), result.end()) != result.end())
    return invalid("matching H terminal row repeats a transport endpoint");
  if (result.empty())
    return invalid("matching service terminal row is empty");
  return result;
}

llvm::Error loom::pnr::detail::verifySystemServiceTargetDomains(
    const FrozenSystemPnrProblem &problem,
    llvm::ArrayRef<PnrIndex> threadChoices,
    llvm::ArrayRef<PnrIndex> graphChoices) {
  for (PnrIndex context = 0; context < problem.serviceContexts().size();
       ++context)
    if (auto domain = resolveSystemServiceTargetDomain(
            problem, context, threadChoices, graphChoices);
        !domain)
      return domain.takeError();
  return llvm::Error::success();
}
