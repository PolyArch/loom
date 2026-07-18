#include "MemoryRealizationProjection.h"

#include "VerifierInternal.h"

#include <algorithm>
#include <tuple>
#include <utility>

using namespace loom::mapping;
using namespace loom::mapping::detail;

llvm::Expected<std::vector<ValidatedMemoryRealizationProjection>>
loom::mapping::detail::buildMemoryRealizationProjections(
    llvm::ArrayRef<MemoryRealizationDraft> realizations,
    const std::map<std::uint64_t, const MemorySemanticEncodingDescriptor *>
        &encodings,
    const std::map<std::uint64_t, const MemoryImplementationDescriptor *>
        &implementations,
    const std::map<std::uint64_t, const MemoryOperationPortTemplateDescriptor *>
        &operationTemplates) {
  std::vector<ValidatedMemoryRealizationProjection> projections;
  projections.reserve(realizations.size());
  for (const MemoryRealizationDraft &realization : realizations) {
    const auto encoding = encodings.find(realization.encoding.entity.value());
    if (encoding == encodings.end())
      return mappingError(MappingErrorCode::InternalError,
                          "validated memory encoding projection is missing");
    const auto implementation =
        implementations.find(encoding->second->implementation.value());
    if (implementation == implementations.end())
      return mappingError(
          MappingErrorCode::InternalError,
          "validated memory implementation projection is missing");

    ValidatedMemoryRealizationProjection projected{
        realization.id,
        encoding->second->id,
        implementation->second->id,
        implementation->second->service,
        {}};
    projected.activeBoundaryPorts.reserve(realization.boundaryPorts.size());
    for (const MemoryBoundaryPortCorrespondence &boundary :
         realization.boundaryPorts) {
      const auto operation = operationTemplates.find(
          boundary.operationPort.operation.entity.value());
      if (operation == operationTemplates.end() ||
          boundary.operationPort.index >= operation->second->ports.size())
        return mappingError(MappingErrorCode::InternalError,
                            "validated memory operation projection is missing");
      const MemoryOperationPortDescriptor &port =
          operation->second->ports[boundary.operationPort.index];
      projected.activeBoundaryPorts.push_back(
          {operation->second->id, port.direction, boundary.operationPort.index,
           port.port});
    }
    std::sort(projected.activeBoundaryPorts.begin(),
              projected.activeBoundaryPorts.end(),
              [](const ValidatedMemoryBoundaryPort &lhs,
                 const ValidatedMemoryBoundaryPort &rhs) {
                return std::make_tuple(lhs.operation.value(), lhs.direction,
                                       lhs.port) <
                       std::make_tuple(rhs.operation.value(), rhs.direction,
                                       rhs.port);
              });
    projections.push_back(std::move(projected));
  }
  std::sort(projections.begin(), projections.end(),
            [](const ValidatedMemoryRealizationProjection &lhs,
               const ValidatedMemoryRealizationProjection &rhs) {
              return lhs.id.value() < rhs.id.value();
            });
  return projections;
}
