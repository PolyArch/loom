#include "Hardware/RTL/Providers/Native/DesignWare.h"

#include "../ProviderSupport.h"
#include "Hardware/RTL/OperationLeaf.h"

#include "Fabric/IR/OperationResourceContract.h"
#include "Fabric/IR/ResourceContractRecord.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace loom::hardware::rtl {
namespace {

std::vector<std::uint8_t> blackBoxBytes() {
  const llvm::ArrayRef<std::uint8_t> bytes =
      synopsysDesignWareDwFpMacBlackBoxContractBytes();
  return {bytes.begin(), bytes.end()};
}

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "synopsys_designware_invalid: " + message);
}

llvm::Error unsupported(const FabricOperationProviderRequest &request) {
  return llvm::make_error<FabricOperationProviderUnsupportedError>(
      request.capability.implementationFamily, request.recipe);
}

bool isExactProfile(const ::fabric::ScalarFloatParams &parameters) {
  const ::fabric::FloatBehaviorProfile &behavior = parameters.behavior;
  return parameters.formats.valid() && parameters.formats.size() == 1 &&
         parameters.formats.contains(::fabric::FloatFormat::F32) &&
         behavior.roundingModes.valid() && behavior.roundingModes.size() == 1 &&
         behavior.roundingModes.contains(
             mlir::arith::RoundingMode::to_nearest_even) &&
         behavior.nanBehaviors.valid() && behavior.nanBehaviors.size() == 1 &&
         behavior.nanBehaviors.contains(::fabric::FloatNaNBehavior::IEEE) &&
         behavior.subnormalBehaviors.valid() &&
         behavior.subnormalBehaviors.size() == 1 &&
         behavior.subnormalBehaviors.contains(
             ::fabric::FloatSubnormalBehavior::Preserve) &&
         behavior.signedZeroBehaviors.valid() &&
         behavior.signedZeroBehaviors.size() == 1 &&
         behavior.signedZeroBehaviors.contains(
             ::fabric::FloatSignedZeroBehavior::Preserve) &&
         behavior.requiredFastMath == mlir::arith::FastMathFlags::none;
}

bool isExactActor(const ::dataflow::CanonicalActorSchemaProjection &actor) {
  if (actor.schema != ::dataflow::OperationSchemaId::MathFma ||
      actor.type.getNumInputs() != 3 || actor.type.getNumResults() != 1)
    return false;
  const mlir::Type type = actor.type.getInput(0);
  if (!type.isF32() || actor.type.getInput(1) != type ||
      actor.type.getInput(2) != type || actor.type.getResult(0) != type)
    return false;
  const auto *payload =
      std::get_if<::dataflow::FloatingPointPayload>(&actor.payload);
  return payload && payload->flags == mlir::arith::FastMathFlags::none &&
         (!payload->roundingMode ||
          *payload->roundingMode == mlir::arith::RoundingMode::to_nearest_even);
}

llvm::Expected<std::vector<const fabric::ResolvedFabricOpPhysicalPortView *>>
exactPorts(const FabricOperationProviderRequest &request,
           fabric::FabricPortDirection direction, unsigned count) {
  std::vector<const fabric::ResolvedFabricOpPhysicalPortView *> ports;
  for (const fabric::ResolvedFabricOpPhysicalPortView &port :
       request.capability.physicalPorts)
    if (port.reference.direction == direction)
      ports.push_back(&port);
  llvm::sort(ports, [](const auto *lhs, const auto *rhs) {
    return lhs->reference.ordinal < rhs->reference.ordinal;
  });
  if (ports.size() != count)
    return unsupported(request);
  for (unsigned ordinal = 0; ordinal != count; ++ordinal)
    if (ports[ordinal]->reference.ordinal != ordinal ||
        ports[ordinal]->payloadWidthBits != 64)
      return unsupported(request);
  return ports;
}

llvm::Expected<FabricOperationProviderOutput>
materializeScalarFloatFma(FabricOperationProviderRequest request) {
  if (request.recipe != BackendRecipeKey::SynopsysDesignWare)
    return invalid("provider received a different recipe");
  if (request.capability.implementationFamily !=
      ::fabric::ImplementationFamilyId::ScalarFloatFma)
    return invalid("provider received a different implementation family");
  if (request.externalImplementationContractRef !=
      synopsysDesignWareContractRef)
    return invalid("provider received a different external contract");
  if (!isSynopsysDesignWareDwFpMacComponentInput(request.externalInputs))
    return unsupported(request);

  const auto *parameters = std::get_if<::fabric::ScalarFloatParams>(
      &request.capability.parameterizedCapability);
  if (!parameters || !isExactProfile(*parameters) ||
      request.capability.enabledOperationSchemas !=
          std::vector<::dataflow::OperationSchemaId>{
              ::dataflow::OperationSchemaId::MathFma} ||
      !request.capability.configurationFieldSchema.empty())
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

  auto inputs = exactPorts(request, fabric::FabricPortDirection::Input, 3);
  if (!inputs)
    return inputs.takeError();
  auto outputs = exactPorts(request, fabric::FabricPortDirection::Output, 1);
  if (!outputs)
    return outputs.takeError();
  if (llvm::Error error = verifyFabricOperationLeafPorts(
          request.leaf, request.occurrence, request.capability,
          request.configurationAbi))
    return std::move(error);

  auto relation = request.capability.resolveSemanticFieldRelation(
      *request.leaf.getContext());
  if (!relation)
    return relation.takeError();
  const auto &domain = relation->finiteBehaviorDomain();
  if (relation->kind() != ::fabric::FabricOpSemanticFieldRelationKind::None ||
      domain.size() != 1 || domain.front().semanticConfiguration ||
      !isExactActor(domain.front().representativeActor))
    return unsupported(request);

  mlir::OpBuilder builder(request.leaf.getContext());
  builder.setInsertionPoint(request.leaf);
  const mlir::Location location = request.leaf.getLoc();
  circt::hw::HWModuleOp::create(
      builder, location, request.leaf.getSymNameAttr(),
      circt::hw::ModulePortInfo(request.leaf.getPortList()),
      [&](mlir::OpBuilder &bodyBuilder,
          circt::hw::HWModulePortAccessor &accessor) {
        llvm::SmallVector<mlir::Value, 5> substitutions;
        for (unsigned index = 0; index != 3; ++index)
          substitutions.push_back(detail::resizeUnsigned(
              bodyBuilder, location,
              accessor.getInput("data_input_" + std::to_string(index)), 32));

        auto resultWire = circt::sv::WireOp::create(
            bodyBuilder, location, bodyBuilder.getI32Type(), "dw_result");
        auto statusWire = circt::sv::WireOp::create(
            bodyBuilder, location, bodyBuilder.getI8Type(), "dw_status");
        substitutions.push_back(resultWire);
        substitutions.push_back(statusWire);
        circt::sv::VerbatimOp::create(
            bodyBuilder, location,
            bodyBuilder.getStringAttr("DW_fp_mac #(\n"
                                      "  .sig_width(23),\n"
                                      "  .exp_width(8),\n"
                                      "  .ieee_compliance(1)\n"
                                      ") designware_component (\n"
                                      "  .a({{0}}),\n"
                                      "  .b({{1}}),\n"
                                      "  .c({{2}}),\n"
                                      "  .rnd(3'b000),\n"
                                      "  .z({{3}}),\n"
                                      "  .status({{4}})\n"
                                      ");"),
            substitutions, bodyBuilder.getArrayAttr({}));
        mlir::Value result =
            circt::sv::ReadInOutOp::create(bodyBuilder, location, resultWire);
        accessor.setOutput(
            "data_output_0",
            detail::resizeUnsigned(bodyBuilder, location, result,
                                   outputs->front()->payloadWidthBits));
      },
      request.leaf.getParametersAttr());
  request.leaf.erase();

  FabricOperationProviderOutput output;
  output.payloads.push_back({PayloadRole::BlackBoxContract,
                             synopsysDesignWareDwFpMacBlackBoxLogicalName.str(),
                             blackBoxBytes()});
  output.externalImplementationBindings.push_back(
      {synopsysDesignWareContractRef.str(),
       std::vector<ExternalInputBinding>(request.externalInputs.begin(),
                                         request.externalInputs.end()),
       {},
       {{RepresentationObjectKind::Module,
         synopsysDesignWareDwFpMacComponentName.str()}},
       ImplementationPayloadKey{
           PayloadRole::BlackBoxContract,
           synopsysDesignWareDwFpMacBlackBoxLogicalName.str()}});
  return output;
}

} // namespace

llvm::Error registerSynopsysDesignWareScalarFloatFmaProvider(
    FabricOperationProviderRegistry &registry) {
  return registry.add({::fabric::ImplementationFamilyId::ScalarFloatFma,
                       BackendRecipeKey::SynopsysDesignWare,
                       synopsysDesignWareContractRef.str(),
                       materializeScalarFloatFma});
}

} // namespace loom::hardware::rtl
