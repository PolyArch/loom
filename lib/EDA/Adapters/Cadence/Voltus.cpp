#include "EDA/Adapters/Cadence/Voltus.h"

#include "llvm/Support/JSON.h"

namespace loom::eda::cadence {
namespace {

constexpr CadenceImplementationState acceptedStates[]{
    {hardware::RepresentationRootVariant::AsicPhysical,
     hardware::RepresentationPhysicalStage::Routed},
};
constexpr llvm::StringLiteral providerInputs[]{"power_grid_library"};
constexpr llvm::StringLiteral declaredOutputs[]{
    "outputs/voltus-rail-result.json"};

const CadenceInvocationDescriptor descriptor{
    "voltus",
    "loom.eda.cadence.voltus.rail@1",
    CadenceOperation::RailEvaluation,
    acceptedStates,
    true,
    true,
    true,
    providerInputs,
    declaredOutputs,
};

llvm::Error parserError(const llvm::Twine &detail) {
  return makeCadenceAdapterError(CadenceAdapterFailureKind::ParserFailure,
                                 descriptor.implementationSemanticIdentity,
                                 detail);
}

} // namespace

const CadenceInvocationDescriptor &voltusRailDescriptor() { return descriptor; }

llvm::Expected<VoltusRailObservation>
parseVoltusRailObservation(llvm::StringRef contents) {
  auto parsed = llvm::json::parse(contents);
  if (!parsed)
    return parserError("rail result is malformed JSON: " +
                       llvm::toString(parsed.takeError()));
  const llvm::json::Object *object = parsed->getAsObject();
  if (!object || object->size() != 3)
    return parserError("rail result shape is invalid");
  const auto schema = object->getString("schema");
  const auto version = object->getString("version");
  const auto maximum = object->getString("maximum_voltage_drop_volts");
  if (!schema || *schema != "loom.cadence.voltus_rail_result" || !version ||
      *version != "1.0" || !maximum)
    return parserError("rail result fields are invalid");
  auto parsedMaximum =
      parseCadenceDecimal(descriptor.implementationSemanticIdentity,
                          "maximum_voltage_drop_volts", *maximum, true);
  if (!parsedMaximum)
    return parsedMaximum.takeError();
  return VoltusRailObservation{*parsedMaximum};
}

llvm::Expected<external_tool::ExternalToolInvocationBundleSpec>
makeVoltusRailBundleSpec(const CadenceBundleInputs &inputs) {
  if (inputs.semanticContract.providerIdentity !=
      descriptor.implementationSemanticIdentity)
    return makeCadenceAdapterError(
        CadenceAdapterFailureKind::DescriptorMismatch,
        descriptor.implementationSemanticIdentity,
        "semantic contract provider does not match the adapter");
  if (!inputs.implementation)
    return makeCadenceAdapterError(
        CadenceAdapterFailureKind::MissingSemanticInput,
        descriptor.implementationSemanticIdentity,
        "exact HardwareImplementation representation is absent");
  if (llvm::Error error =
          validateCadenceRepresentation(descriptor, *inputs.implementation))
    return std::move(error);
  return makeCadenceAdapterError(
      CadenceAdapterFailureKind::MissingProviderInput,
      descriptor.implementationSemanticIdentity,
      "ExternalTool has no directory-valued fingerprint for the complete "
      "Voltus power_grid_library closure");
}

} // namespace loom::eda::cadence
