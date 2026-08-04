#include "FabricModuleBoundaryTransportValidation.h"

#include "FabricArtifactViewInternal.h"

#include "Fabric/Identity/FabricRefBytes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <set>
#include <vector>

namespace {

using namespace loom::fabric;

llvm::Error invalidView(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "fabric_artifact_invalid: " + message);
}

const loom::fabric::detail::FabricModuleBoundaryEndpointViewData *
boundaryRecord(const loom::fabric::detail::FabricArtifactViewData &data,
               const FabricModuleBoundaryEndpointRef &reference) {
  if (reference.module.id() >= data.entities.size())
    return nullptr;
  const auto &module = data.entities[reference.module.id()];
  if (module.kind != FabricEntityKind::FabricModuleTemplate)
    return nullptr;
  const auto *records = reference.direction == FabricPortDirection::Input
                            ? &module.moduleBoundaryInputs
                            : &module.moduleBoundaryOutputs;
  return reference.ordinal < records->size() ? &(*records)[reference.ordinal]
                                             : nullptr;
}

} // namespace

bool loom::fabric::detail::haveSameFabricTransportKind(
    llvm::ArrayRef<std::uint8_t> left, llvm::ArrayRef<std::uint8_t> right) {
  constexpr std::size_t kindBytes = sizeof(std::uint32_t);
  return left.size() >= kindBytes && right.size() >= kindBytes &&
         left.take_front(kindBytes) == right.take_front(kindBytes);
}

llvm::Error
loom::fabric::detail::canonicalizeFabricModuleBoundaryTransportRelations(
    FabricArtifactViewData &data) {
  if (data.rootKind != FabricRootKind::Module) {
    if (!data.moduleBoundaryTransportAttachments.empty() ||
        !data.moduleBoundaryTransportPassthroughs.empty())
      return invalidView(
          "only a Module root may expose boundary transport relations");
    return llvm::Error::success();
  }

  llvm::sort(data.moduleBoundaryTransportAttachments,
             [](const auto &lhs, const auto &rhs) {
               const auto lhsBoundary = canonicalFabricBytes(lhs.boundary);
               const auto rhsBoundary = canonicalFabricBytes(rhs.boundary);
               if (lhsBoundary != rhsBoundary)
                 return lhsBoundary < rhsBoundary;
               return canonicalFabricBytes(lhs.endpoint) <
                      canonicalFabricBytes(rhs.endpoint);
             });
  llvm::sort(data.moduleBoundaryTransportPassthroughs, [](const auto &lhs,
                                                          const auto &rhs) {
    const auto lhsOutput = canonicalFabricBytes(lhs.output);
    const auto rhsOutput = canonicalFabricBytes(rhs.output);
    if (lhsOutput != rhsOutput)
      return lhsOutput < rhsOutput;
    return canonicalFabricBytes(lhs.input) < canonicalFabricBytes(rhs.input);
  });

  std::set<std::vector<std::uint8_t>> connectedBoundaries;
  for (const FabricModuleBoundaryTransportAttachmentView &attachment :
       data.moduleBoundaryTransportAttachments) {
    const auto *record = boundaryRecord(data, attachment.boundary);
    if (!record ||
        record->plane != FabricSpatialAttachmentEndpointRef::Plane::Transport)
      return invalidView(
          "an invalid Module boundary has a transport attachment");
    if (!connectedBoundaries.insert(canonicalFabricBytes(attachment.boundary))
             .second)
      return invalidView(
          "a Module boundary has more than one transport relation");
  }

  for (const FabricModuleBoundaryTransportPassthroughView &passthrough :
       data.moduleBoundaryTransportPassthroughs) {
    const auto *input = boundaryRecord(data, passthrough.input);
    const auto *output = boundaryRecord(data, passthrough.output);
    if (passthrough.input.direction != FabricPortDirection::Input ||
        passthrough.output.direction != FabricPortDirection::Output ||
        passthrough.input.module != passthrough.output.module || !input ||
        !output ||
        input->plane != FabricSpatialAttachmentEndpointRef::Plane::Transport ||
        output->plane != FabricSpatialAttachmentEndpointRef::Plane::Transport)
      return invalidView("a Module boundary passthrough is not token-plane "
                         "input-to-output correspondence");
    if (!connectedBoundaries.insert(canonicalFabricBytes(passthrough.input))
             .second ||
        !connectedBoundaries.insert(canonicalFabricBytes(passthrough.output))
             .second)
      return invalidView(
          "a Module boundary has more than one transport relation");
  }

  const FabricEntityViewData *module = nullptr;
  FabricModuleTemplateRef moduleRef;
  for (auto [id, entity] : llvm::enumerate(data.entities)) {
    if (entity.kind != FabricEntityKind::FabricModuleTemplate)
      continue;
    if (module)
      return invalidView("Module root owns more than one Module template");
    module = &entity;
    moduleRef = FabricModuleTemplateRef(id);
  }
  if (!module)
    return invalidView("Module root has no Module template");
  for (FabricOrdinal ordinal = 0;
       ordinal < module->moduleBoundaryOutputs.size(); ++ordinal) {
    if (module->moduleBoundaryOutputs[ordinal].plane !=
        FabricSpatialAttachmentEndpointRef::Plane::Transport)
      continue;
    const FabricModuleBoundaryEndpointRef output{
        moduleRef, FabricPortDirection::Output, ordinal};
    if (!connectedBoundaries.count(canonicalFabricBytes(output)))
      return invalidView("a Module token output has no transport relation");
  }
  return llvm::Error::success();
}

llvm::Error
loom::fabric::detail::validateFabricModuleBoundaryTransportRelations(
    const FabricArtifactView &view) {
  std::set<std::vector<std::uint8_t>> attachedTransportEndpoints;
  for (const FabricModuleBoundaryTransportAttachmentView &attachment :
       view.moduleBoundaryTransportAttachments()) {
    if (llvm::Error error = validateFabricRef(view, attachment.boundary))
      return error;
    if (llvm::Error error = validateFabricRef(view, attachment.endpoint))
      return error;
    if (view.moduleBoundaryEndpointPlane(attachment.boundary) !=
        FabricSpatialAttachmentEndpointRef::Plane::Transport)
      return invalidView("a memory Module boundary has a transport attachment");
    if (view.transportEndpointDirection(attachment.endpoint) !=
        attachment.boundary.direction)
      return invalidView("a Module boundary attachment changes direction");
    if (!haveSameFabricTransportKind(
            view.moduleBoundaryEndpointType(attachment.boundary),
            view.transportEndpointType(attachment.endpoint)))
      return invalidView("a Module boundary attachment changes transport kind");
    if (!attachedTransportEndpoints
             .insert(canonicalFabricBytes(attachment.endpoint))
             .second)
      return invalidView(
          "an occurrence endpoint is attached to multiple Module boundaries");
  }

  for (const FabricModuleBoundaryTransportPassthroughView &passthrough :
       view.moduleBoundaryTransportPassthroughs()) {
    if (llvm::Error error = validateFabricRef(view, passthrough.input))
      return error;
    if (llvm::Error error = validateFabricRef(view, passthrough.output))
      return error;
    if (passthrough.input.direction != FabricPortDirection::Input ||
        passthrough.output.direction != FabricPortDirection::Output ||
        passthrough.input.module != passthrough.output.module)
      return invalidView(
          "a Module boundary passthrough changes owner or direction");
    if (view.moduleBoundaryEndpointPlane(passthrough.input) !=
            FabricSpatialAttachmentEndpointRef::Plane::Transport ||
        view.moduleBoundaryEndpointPlane(passthrough.output) !=
            FabricSpatialAttachmentEndpointRef::Plane::Transport)
      return invalidView("a memory Module boundary has a token passthrough");
    if (!haveSameFabricTransportKind(
            view.moduleBoundaryEndpointType(passthrough.input),
            view.moduleBoundaryEndpointType(passthrough.output)))
      return invalidView(
          "a Module boundary passthrough changes transport kind");
  }
  return llvm::Error::success();
}
