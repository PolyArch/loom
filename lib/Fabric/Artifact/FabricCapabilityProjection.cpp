#include "FabricCapabilityProjection.h"

#include "Fabric/IR/ImplementationFamily.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>
#include <string>
#include <utility>

using namespace mlir;

namespace loom::fabric::detail {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "fabric_artifact_invalid: " + message);
}

} // namespace

llvm::Expected<std::vector<std::uint8_t>> projectMemoryEndpointType(Type type) {
  if (!isa<MemRefType>(type))
    return invalid("memory endpoint has a non-memref physical type");
  std::string spelling;
  llvm::raw_string_ostream stream(spelling);
  type.print(stream);
  return std::vector<std::uint8_t>(spelling.begin(), spelling.end());
}

llvm::Error setModuleBoundaryInventory(::fabric::ModuleOp root,
                                       FabricEntityViewData &entity) {
  FunctionType type = root.getFunctionType();
  entity.owner.inventoryCounts[static_cast<std::size_t>(
      FabricInventoryKind::InputPort)] = type.getNumInputs();
  entity.owner.inventoryCounts[static_cast<std::size_t>(
      FabricInventoryKind::OutputPort)] = type.getNumResults();

  std::uint64_t tokenInput = 0;
  std::uint64_t memoryInput = 0;
  std::uint64_t tokenOutput = 0;
  std::uint64_t memoryOutput = 0;
  auto append = [&](Type endpointType, bool input) -> llvm::Error {
    const bool memory = isa<MemRefType>(endpointType);
    auto encoded = memory ? projectMemoryEndpointType(endpointType)
                          : ::fabric::encodeFabricTransportType(endpointType);
    if (!encoded)
      return encoded.takeError();
    FabricModuleBoundaryEndpointViewData endpoint{
        memory ? FabricSpatialAttachmentEndpointRef::Plane::Memory
               : FabricSpatialAttachmentEndpointRef::Plane::Transport,
        0, std::move(*encoded)};
    if (input) {
      endpoint.occurrenceOrdinal = memory ? memoryInput++ : tokenInput++;
      entity.moduleBoundaryInputs.push_back(std::move(endpoint));
    } else {
      endpoint.occurrenceOrdinal = memory ? memoryOutput++ : tokenOutput++;
      entity.moduleBoundaryOutputs.push_back(std::move(endpoint));
    }
    return llvm::Error::success();
  };
  for (Type input : type.getInputs())
    if (llvm::Error error = append(input, true))
      return error;
  for (Type output : type.getResults())
    if (llvm::Error error = append(output, false))
      return error;

  for (FabricModuleBoundaryEndpointViewData &endpoint :
       entity.moduleBoundaryOutputs)
    endpoint.occurrenceOrdinal +=
        endpoint.plane == FabricSpatialAttachmentEndpointRef::Plane::Memory
            ? memoryInput
            : tokenInput;
  return llvm::Error::success();
}

llvm::Expected<ResolvedFabricOpCapabilityView>
resolveFabricOpCapability(::fabric::OpOp operation,
                          const FabricFuTemplateNodeRef &reference,
                          FabricFuNodeViewData &node) {
  std::optional<::fabric::ImplementationFamilyId> family =
      operation.getImplementationFamily();
  if (!family)
    return invalid("fabric.op has no implementation family");

  auto parameters =
      ::fabric::parseFamilyCapabilityParams(*family, operation.getHwParams());
  if (!parameters)
    return parameters.takeError();

  std::vector<::dataflow::OperationSchemaId> enabledSchemas;
  enabledSchemas.reserve(operation.getOpList().size());
  for (Attribute attribute : operation.getOpList()) {
    auto symbol = dyn_cast<FlatSymbolRefAttr>(attribute);
    if (!symbol)
      return invalid("fabric.op has a malformed operation schema member");
    std::optional<::dataflow::OperationSchemaId> schema =
        ::dataflow::findOperationSchema(symbol.getValue());
    if (!schema)
      return invalid("fabric.op names an unregistered operation schema");
    enabledSchemas.push_back(*schema);
  }

  const bool enablesGetElementPtr = llvm::is_contained(
      enabledSchemas, ::dataflow::OperationSchemaId::LLVMGetElementPtr);
  const auto *integerParameters =
      std::get_if<::fabric::ScalarIntegerParams>(&*parameters);
  const bool hasPointerFormats =
      integerParameters && !integerParameters->pointerFormats.empty();
  if (enablesGetElementPtr != hasPointerFormats)
    return invalid("fabric.op GEP member and pointer_formats must be present "
                   "or absent together");
  if (hasPointerFormats &&
      *family != ::fabric::ImplementationFamilyId::ScalarIntegerAddSub)
    return invalid("pointer_formats are unavailable for this implementation "
                   "family");

  if (!node.owner.resourceContract)
    return invalid("fabric.op has no resolved resource contract");

  std::vector<ResolvedFabricOpPhysicalPortView> ports;
  ports.reserve(node.owner.transportEndpoints.size());
  FabricOrdinal inputOrdinal = 0;
  FabricOrdinal outputOrdinal = 0;
  for (auto [portOrdinal, endpoint] :
       llvm::enumerate(node.owner.transportEndpoints)) {
    Type physicalType =
        portOrdinal < operation.getNumOperands()
            ? operation.getOperand(portOrdinal).getType()
            : operation.getResult(portOrdinal - operation.getNumOperands())
                  .getType();
    std::optional<unsigned> payloadWidth =
        ::fabric::getFabricBitsWidth(physicalType);
    if (!payloadWidth)
      return invalid("fabric.op physical port is not untagged Fabric bits");
    FabricOrdinal ordinal = endpoint.direction == FabricPortDirection::Input
                                ? inputOrdinal++
                                : outputOrdinal++;
    ports.push_back(ResolvedFabricOpPhysicalPortView{
        FabricFuNodePortRef{reference, endpoint.direction, ordinal},
        endpoint.canonicalType, *payloadWidth});
  }

  std::vector<std::uint32_t> physicalInputWidths;
  std::vector<std::uint32_t> physicalResultWidths;
  physicalInputWidths.reserve(inputOrdinal);
  physicalResultWidths.reserve(outputOrdinal);
  for (const ResolvedFabricOpPhysicalPortView &port : ports) {
    if (port.reference.direction == FabricPortDirection::Input)
      physicalInputWidths.push_back(port.payloadWidthBits);
    else
      physicalResultWidths.push_back(port.payloadWidthBits);
  }

  auto semanticFieldRelation = ::fabric::resolveFabricOpSemanticFieldRelation(
      *family, *parameters, enabledSchemas, physicalInputWidths,
      physicalResultWidths, *operation.getContext());
  if (!semanticFieldRelation)
    return semanticFieldRelation.takeError();

  std::vector<FabricSemanticConfigFieldRef> configurationFields;
  if (semanticFieldRelation->hasConfigurationField()) {
    node.owner.inventoryCounts[static_cast<std::size_t>(
        FabricInventoryKind::SemanticConfigField)] = 1;
    configurationFields.push_back(FabricSemanticConfigFieldRef{
        FabricConfigurationOwnerRef(FabricInventoryOwnerRef::of(reference)),
        0});
  }

  std::vector<FabricPhysicalRefinementDomainRef> refinements;
  const std::uint64_t refinementCount =
      node.owner.inventoryCounts[static_cast<std::size_t>(
          FabricInventoryKind::RefinementDomain)];
  refinements.reserve(refinementCount);
  for (FabricOrdinal ordinal = 0; ordinal < refinementCount; ++ordinal)
    refinements.push_back(FabricPhysicalRefinementDomainRef{
        FabricRefinementOwnerRef(FabricInventoryOwnerRef::of(reference)),
        ordinal});

  return ResolvedFabricOpCapabilityView{reference,
                                        *family,
                                        std::move(enabledSchemas),
                                        std::move(*parameters),
                                        std::move(ports),
                                        std::move(configurationFields),
                                        *node.owner.resourceContract,
                                        std::move(refinements)};
}

} // namespace loom::fabric::detail
