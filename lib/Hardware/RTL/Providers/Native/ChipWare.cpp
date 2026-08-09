#include "Hardware/RTL/Providers/Native/ChipWare.h"

#include "Hardware/RTL/OperationLeaf.h"

#include "Dataflow/IR/OperationSchema.h"
#include "Fabric/IR/OperationResourceContract.h"
#include "Fabric/IR/ResourceContractRecord.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"

#include <cstdint>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace loom::hardware::rtl {
namespace {

std::string chipWareBlackBoxContract() {
  std::string contract = "component=";
  contract += cadenceChipWareCwMultModuleName;
  contract += "\nresource_key=";
  contract += cadenceChipWareCwMultResourceKey;
  contract += "\nparameters=wA:8,wB:8\n"
              "ports=A:input:8,B:input:8,TC:input:1,Z:output:16\n"
              "mode=TC:0\n"
              "component_latency=combinational\n"
              "fabric_progress=one_cycle_elastic\n"
              "result=Z[7:0]\n";
  return contract;
}

std::vector<std::uint8_t> chipWareBlackBoxBytes() {
  const std::string contract = chipWareBlackBoxContract();
  return {contract.begin(), contract.end()};
}

std::string chipWareInstantiation() {
  std::string instantiation = cadenceChipWareCwMultModuleName.str();
  instantiation += " #(\n"
                   "  .wA(8),\n"
                   "  .wB(8)\n"
                   ") chipware_multiplier (\n"
                   "  .A({{0}}),\n"
                   "  .B({{1}}),\n"
                   "  .TC(1'b0),\n"
                   "  .Z({{2}})\n"
                   ");";
  return instantiation;
}

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      llvm::inconvertibleErrorCode(),
      "cadence_chipware_scalar_integer_multiply_invalid: " + message);
}

llvm::Error unsupported(const FabricOperationProviderRequest &request) {
  return llvm::make_error<FabricOperationProviderUnsupportedError>(
      request.capability.implementationFamily, request.recipe);
}

bool isExactComponentInput(llvm::ArrayRef<ExternalInputBinding> inputs) {
  if (inputs.size() != 1 || inputs.front().providerInputSlotRef !=
                                cadenceChipWareComponentModelSlotRef)
    return false;
  const auto *resource = std::get_if<ToolBundledResourceDependency>(
      &inputs.front().dependencyIdentity);
  return resource && resource->resourceKey == cadenceChipWareCwMultResourceKey;
}

llvm::Expected<const ToolBundledResourceDependency *>
requireComponentModel(const FabricOperationProviderRequest &request) {
  if (request.externalImplementationContractRef !=
      cadenceChipWareExternalContractRef)
    return invalid("provider received a different external contract");
  if (request.externalInputs.size() != 1 ||
      request.externalInputs.front().providerInputSlotRef !=
          cadenceChipWareComponentModelSlotRef)
    return invalid("provider received a malformed component model closure");
  const auto *resource = std::get_if<ToolBundledResourceDependency>(
      &request.externalInputs.front().dependencyIdentity);
  if (!resource)
    return invalid("component model is not a tool-bundled resource");
  return resource;
}

llvm::Error requireExactBehavior(FabricOperationProviderRequest &request) {
  if (request.capability.enabledOperationSchemas !=
      std::vector<::dataflow::OperationSchemaId>{
          ::dataflow::OperationSchemaId::ArithMulI})
    return unsupported(request);
  if (!request.capability.configurationFieldSchema.empty())
    return unsupported(request);

  auto relation = request.capability.resolveSemanticFieldRelation(
      *request.leaf.getContext());
  if (!relation)
    return relation.takeError();
  const auto &domain = relation->finiteBehaviorDomain();
  if (relation->kind() != ::fabric::FabricOpSemanticFieldRelationKind::None ||
      domain.size() != 1 || domain.front().semanticConfiguration)
    return invalid("fixed multiply does not have a singleton behavior");
  const auto &actor = domain.front().representativeActor;
  if (actor.schema != ::dataflow::OperationSchemaId::ArithMulI ||
      actor.type.getNumInputs() != 2 || actor.type.getNumResults() != 1)
    return invalid("fixed multiply behavior has the wrong schema");
  const auto *overflow =
      std::get_if<::dataflow::IntegerOverflowPayload>(&actor.payload);
  if (!overflow)
    return invalid("fixed multiply behavior has the wrong payload");
  if (overflow->flags != mlir::arith::IntegerOverflowFlags::none)
    return unsupported(request);
  auto type = llvm::dyn_cast<mlir::IntegerType>(actor.type.getInput(0));
  if (!type || type.getWidth() != 8 || actor.type.getInput(1) != type ||
      actor.type.getResult(0) != type)
    return unsupported(request);
  return llvm::Error::success();
}

llvm::Expected<FabricOperationProviderOutput>
materializeCadenceChipWareScalarIntegerMultiply(
    FabricOperationProviderRequest request) {
  if (request.recipe != BackendRecipeKey::CadenceChipWare)
    return invalid("provider received a different recipe");
  if (request.capability.implementationFamily !=
      ::fabric::ImplementationFamilyId::ScalarIntegerMultiply)
    return invalid("provider received a different implementation family");
  const auto *parameters = std::get_if<::fabric::ScalarIntegerParams>(
      &request.capability.parameterizedCapability);
  if (!parameters || !parameters->integerWidths.valid())
    return invalid("capability has the wrong parameter schema");
  if (parameters->integerWidths.size() != 1 ||
      !parameters->integerWidths.contains(::fabric::IntegerWidth::I8) ||
      !parameters->pointerFormats.empty())
    return unsupported(request);

  auto componentModel = requireComponentModel(request);
  if (!componentModel)
    return componentModel.takeError();
  if ((*componentModel)->resourceKey != cadenceChipWareCwMultResourceKey)
    return unsupported(request);

  auto actualContract = ::fabric::encodeResourceContractRecord(
      request.capability.resourceStateAndTimingContract);
  if (!actualContract)
    return actualContract.takeError();
  auto supportedContract = ::fabric::encodeResourceContractRecord(
      ::fabric::oneCycleElasticOperationResourceContract());
  if (!supportedContract)
    return supportedContract.takeError();
  if (*actualContract != *supportedContract)
    return unsupported(request);

  std::vector<const fabric::ResolvedFabricOpPhysicalPortView *> inputs;
  std::vector<const fabric::ResolvedFabricOpPhysicalPortView *> outputs;
  for (const fabric::ResolvedFabricOpPhysicalPortView &port :
       request.capability.physicalPorts)
    (port.reference.direction == fabric::FabricPortDirection::Input ? inputs
                                                                    : outputs)
        .push_back(&port);
  const auto byOrdinal = [](const auto *lhs, const auto *rhs) {
    return lhs->reference.ordinal < rhs->reference.ordinal;
  };
  llvm::sort(inputs, byOrdinal);
  llvm::sort(outputs, byOrdinal);
  if (inputs.size() != 2 || outputs.size() != 1 ||
      inputs[0]->reference.ordinal != 0 || inputs[1]->reference.ordinal != 1 ||
      outputs[0]->reference.ordinal != 0 || inputs[0]->payloadWidthBits != 8 ||
      inputs[1]->payloadWidthBits != 8 || outputs[0]->payloadWidthBits != 8)
    return unsupported(request);
  if (llvm::Error error = verifyFabricOperationLeafPorts(
          request.leaf, request.occurrence, request.capability,
          request.configurationAbi))
    return std::move(error);
  if (llvm::Error error = requireExactBehavior(request))
    return std::move(error);

  const std::string instantiation = chipWareInstantiation();
  mlir::OpBuilder builder(request.leaf.getContext());
  builder.setInsertionPoint(request.leaf);
  const mlir::Location location = request.leaf.getLoc();
  circt::hw::HWModuleOp::create(
      builder, location, request.leaf.getSymNameAttr(),
      circt::hw::ModulePortInfo(request.leaf.getPortList()),
      [&](mlir::OpBuilder &bodyBuilder,
          circt::hw::HWModulePortAccessor &accessor) {
        circt::sv::WireOp product = circt::sv::WireOp::create(
            bodyBuilder, location, bodyBuilder.getIntegerType(16),
            "chipware_product");
        llvm::SmallVector<mlir::Value, 3> substitutions{
            accessor.getInput("data_input_0"),
            accessor.getInput("data_input_1"), product};
        circt::sv::VerbatimOp::create(
            bodyBuilder, location, bodyBuilder.getStringAttr(instantiation),
            substitutions, bodyBuilder.getArrayAttr({}));
        mlir::Value fullProduct =
            circt::sv::ReadInOutOp::create(bodyBuilder, location, product);
        accessor.setOutput("data_output_0",
                           circt::comb::ExtractOp::create(bodyBuilder, location,
                                                          fullProduct, 0, 8));
      },
      request.leaf.getParametersAttr());
  request.leaf.erase();

  const std::string payloadName =
      cadenceChipWareCwMultBlackBoxLogicalName.str();
  const std::vector<std::uint8_t> blackBoxContract = chipWareBlackBoxBytes();
  FabricOperationProviderOutput output;
  output.payloads.push_back(
      {PayloadRole::BlackBoxContract, payloadName, blackBoxContract});
  output.externalImplementationBindings.push_back(
      {cadenceChipWareExternalContractRef.str(),
       std::vector<ExternalInputBinding>(request.externalInputs.begin(),
                                         request.externalInputs.end()),
       {},
       {{RepresentationObjectKind::Module,
         cadenceChipWareCwMultModuleName.str()}},
       ImplementationPayloadKey{PayloadRole::BlackBoxContract, payloadName}});
  return output;
}

llvm::Error
validateChipWareBinding(const ExternalImplementationBindingDraft &binding,
                        const ImplementationRepresentationRoot &representation,
                        const platform::ImplementationPlatform *) {
  const ImplementationPayloadKey expectedPayload{
      PayloadRole::BlackBoxContract,
      cadenceChipWareCwMultBlackBoxLogicalName.str()};
  const ImplementationPayload expectedDescriptor{
      expectedPayload.role, expectedPayload.canonicalLogicalName,
      computeBlobDigest(chipWareBlackBoxBytes())};
  if (representation.variant != RepresentationRootVariant::Rtl ||
      binding.providerContractRef != cadenceChipWareExternalContractRef ||
      !isExactComponentInput(binding.externalInputs) ||
      binding.fabricResourceRefs.empty() ||
      binding.representationLocators !=
          std::vector<RepresentationLocator>{
              {RepresentationObjectKind::Module,
               cadenceChipWareCwMultModuleName.str()}} ||
      !binding.blackBoxContractPayload ||
      !(*binding.blackBoxContractPayload == expectedPayload) ||
      !llvm::is_contained(representation.payloads, expectedDescriptor))
    return invalid("binding does not preserve the verified CW_mult closure");
  return llvm::Error::success();
}

} // namespace

llvm::Error registerCadenceChipWareExternalImplementationContract(
    ExternalImplementationContractCatalog &catalog) {
  return catalog.add(ExternalImplementationContract{
      cadenceChipWareExternalContractRef.str(),
      {{cadenceChipWareComponentModelSlotRef.str(),
        {ExternalDependencyKind::ToolBundledResource}}},
      {RepresentationRootVariant::Rtl},
      true,
      false,
      validateChipWareBinding});
}

llvm::Error registerCadenceChipWareScalarIntegerMultiplyProvider(
    FabricOperationProviderRegistry &registry) {
  return registry.add({::fabric::ImplementationFamilyId::ScalarIntegerMultiply,
                       BackendRecipeKey::CadenceChipWare,
                       cadenceChipWareExternalContractRef.str(),
                       materializeCadenceChipWareScalarIntegerMultiply});
}

} // namespace loom::hardware::rtl
