#include "TemporalPeTagDomainTestSupport.h"

#include "Fabric/IR/ResourceContractRecord.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <cstdint>
#include <vector>

namespace loom::test {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), message);
}

} // namespace

llvm::Error verifyTemporalPeIngressTagDomains(
    const fabric::FinalizedFabricRoot &fabric) {
  const auto domains = fabric.view().physicalTagMatchDomains();
  if (fabric.view().peOccurrences().size() != 1 || domains.size() != 5)
    return invalid("temporal PE did not expose one tag match domain per "
                   "ingress");

  const auto pe = fabric.view().peOccurrences().front();
  const auto owner = fabric::FabricTransportEndpointOwnerRef::of(pe);
  const auto inventoryOwner = fabric::FabricInventoryOwnerRef::of(pe);
  const auto *contract = fabric.view().resourceContract(inventoryOwner);
  if (!contract)
    return invalid("temporal PE has no resource contract");

  std::vector<std::uint64_t> observed;
  std::uint32_t ingressAssignments = 0;
  std::uint32_t writerAssignments = 0;
  for (std::uint64_t ordinal = 0;
       ordinal < fabric.view().transportEndpointCount(owner); ++ordinal) {
    const fabric::FabricTransportEndpointRef endpoint{owner, ordinal};
    const auto domain = fabric.view().transportEndpointTagMatchDomain(endpoint);
    const auto assignment = fabric.view().physicalTagAssignmentPoint(endpoint);
    if (!assignment)
      return invalid("temporal PE tagged endpoint has no assignment pattern");
    if (assignment->pattern.owner.catalog() != inventoryOwner ||
        assignment->pattern.ordinal >= contract->usePatternCount())
      return invalid("temporal PE assignment pattern has the wrong owner");
    const auto assignmentPattern =
        contract->usePattern(::fabric::UsePatternKey(
            assignment->pattern.ordinal));
    if (!assignmentPattern.claims.empty() || assignmentPattern.commit ||
        !assignmentPattern.parameters.empty() ||
        assignmentPattern.sharingAssignments !=
            llvm::ArrayRef<::fabric::UsePatternValueSchema>(
                {::fabric::UsePatternValueSchema::physicalTag(4)}))
      return invalid("temporal PE assignment pattern is not owner-exact");

    if (fabric.view().transportEndpointDirection(endpoint) ==
        fabric::FabricPortDirection::Input) {
      if (assignment->kind !=
          fabric::FabricPhysicalTagAssignmentPointKind::Ingress)
        return invalid("temporal PE ingress acquired a writer assignment");
      ++ingressAssignments;
      if (!domain || *domain >= domains.size())
        return invalid("temporal PE ingress has no tag match domain");
      const auto &record = domains[*domain];
      if (record.kind !=
              fabric::FabricPhysicalTagMatchDomainKind::TemporalPeIngress ||
          record.owner != inventoryOwner || record.ingress != endpoint ||
          record.tagWidthBits != 4)
        return invalid(
            "temporal PE ingress tag match domain changed owner or width");
      observed.push_back(*domain);
    } else {
      if (assignment->kind !=
          fabric::FabricPhysicalTagAssignmentPointKind::Writer)
        return invalid("temporal PE output acquired an ingress assignment");
      ++writerAssignments;
      if (domain)
        return invalid("temporal PE output became a tag match domain");
    }
  }
  if (ingressAssignments != 5 || writerAssignments != 4)
    return invalid("temporal PE assignment-point inventory changed shape");
  llvm::sort(observed);
  if (std::adjacent_find(observed.begin(), observed.end()) != observed.end())
    return invalid("two temporal PE ingresses share a tag match domain");
  return llvm::Error::success();
}

} // namespace loom::test
