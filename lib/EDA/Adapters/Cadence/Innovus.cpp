#include "EDA/Adapters/Cadence/Innovus.h"

#include "EDA/Adapters/Cadence/Genus.h"
#include "Hardware/Implementation/PhysicalRepresentationIndex.h"
#include "Hardware/Implementation/RepresentationIndex.h"

#include "llvm/ADT/STLExtras.h"

#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace loom::eda::cadence {
namespace {

constexpr CadenceImplementationState acceptedStates[]{
    {hardware::RepresentationRootVariant::GateNetlist, std::nullopt}};
constexpr llvm::StringLiteral providerInputs[]{
    "cell_lef", "qrc_technology_file", "technology_lef", "timing_liberty"};
constexpr llvm::StringLiteral declaredOutputs[]{
    "outputs/innovus-routed.v",
    "outputs/innovus-routed.def",
    "outputs/innovus-routed.sdc",
};

const CadenceInvocationDescriptor descriptor{
    &external_tool::innovusProvider(),
    "loom.eda.cadence.innovus.asic_physical@1",
    CadenceOperation::PhysicalImplementation,
    acceptedStates,
    true,
    true,
    true,
    providerInputs,
    declaredOutputs,
};

const external_tool::ResolvedExternalFile *
findExternal(const CadenceBundleInputs &inputs, llvm::StringRef slot) {
  const auto found =
      llvm::find_if(inputs.frozen.externalFiles, [&](const auto &file) {
        return file.providerInputSlot == slot;
      });
  return found == inputs.frozen.externalFiles.end() ? nullptr : &*found;
}

llvm::Expected<std::string> checkedWord(llvm::StringRef value,
                                        bool bundleInput) {
  if (bundleInput)
    if (llvm::Error error = validateBundleInputPath(
            descriptor.implementationSemanticIdentity, value))
      return std::move(error);
  return renderTclWord(descriptor.implementationSemanticIdentity, value);
}

} // namespace

const CadenceInvocationDescriptor &innovusDescriptor() { return descriptor; }

llvm::Expected<std::string> renderInnovusDriver(llvm::StringRef top,
                                                llvm::StringRef gateNetlist,
                                                llvm::StringRef floorplan,
                                                llvm::StringRef technologyLef,
                                                llvm::StringRef cellLef) {
  if (!isPortableHdlIdentifier(top))
    return makeCadenceAdapterError(
        CadenceAdapterFailureKind::MissingSemanticInput,
        descriptor.implementationSemanticIdentity,
        "top is not a portable HDL identifier");
  auto topWord = checkedWord(top, false);
  if (!topWord)
    return topWord.takeError();
  auto netlistWord = checkedWord(gateNetlist, true);
  if (!netlistWord)
    return netlistWord.takeError();
  auto floorplanWord = checkedWord(floorplan, true);
  if (!floorplanWord)
    return floorplanWord.takeError();
  auto technologyLefWord = checkedWord(technologyLef, false);
  if (!technologyLefWord)
    return technologyLefWord.takeError();
  auto cellLefWord = checkedWord(cellLef, false);
  if (!cellLefWord)
    return cellLefWord.takeError();
  const std::string commands =
      "global init_top_cell init_verilog init_lef_file init_mmmc_file\n"
      "set init_top_cell " +
      *topWord +
      "\n"
      "set init_verilog [list " +
      *netlistWord +
      "]\n"
      "set init_lef_file [list " +
      *technologyLefWord + " " + *cellLefWord +
      "]\n"
      "set init_mmmc_file {drivers/innovus-mmmc.tcl}\n"
      "init_design\n"
      "defIn " +
      *floorplanWord +
      "\n"
      "place_design\n"
      "ccopt_design\n"
      "routeDesign\n";
  return renderCadenceTclBatch(commands,
                               "saveNetlist {outputs/innovus-routed.v}\n"
                               "defOut -routing "
                               "{outputs/innovus-routed.def}\n"
                               "write_sdc {outputs/innovus-routed.sdc}\n");
}

llvm::Expected<std::string>
renderInnovusMmmcDriver(llvm::StringRef generationConstraint,
                        llvm::StringRef timingLiberty,
                        llvm::StringRef qrcTechnologyFile) {
  auto constraintWord = checkedWord(generationConstraint, true);
  if (!constraintWord)
    return constraintWord.takeError();
  auto timingLibertyWord = checkedWord(timingLiberty, false);
  if (!timingLibertyWord)
    return timingLibertyWord.takeError();
  auto qrcTechnologyWord = checkedWord(qrcTechnologyFile, false);
  if (!qrcTechnologyWord)
    return qrcTechnologyWord.takeError();
  return "create_library_set -name {loom_library} -timing [list " +
         *timingLibertyWord +
         "]\n"
         "create_rc_corner -name {loom_rc} -qx_tech_file " +
         *qrcTechnologyWord +
         "\n"
         "create_delay_corner -name {loom_delay} -library_set {loom_library} "
         "-rc_corner {loom_rc}\n"
         "create_constraint_mode -name {loom_constraints} -sdc_files [list " +
         *constraintWord +
         "]\n"
         "create_analysis_view -name {loom_view} -constraint_mode "
         "{loom_constraints} -delay_corner {loom_delay}\n"
         "set_analysis_view -setup [list {loom_view}] -hold [list "
         "{loom_view}]\n";
}

llvm::Expected<InnovusPhysicalSnapshot> parseInnovusPhysicalSnapshot(
    llvm::StringRef netlist, llvm::StringRef designExchangeFormat,
    llvm::StringRef generationConstraints, llvm::StringRef top,
    hardware::RepresentationPhysicalStage stage) {
  if (stage != hardware::RepresentationPhysicalStage::Routed)
    return makeCadenceAdapterError(
        CadenceAdapterFailureKind::UnsupportedImplementation,
        descriptor.implementationSemanticIdentity,
        "Innovus route driver publishes only Routed state");
  auto parsedNetlist = parseGenusGateNetlist(netlist, top);
  if (!parsedNetlist)
    return makeCadenceAdapterError(CadenceAdapterFailureKind::ParserFailure,
                                   descriptor.implementationSemanticIdentity,
                                   llvm::toString(parsedNetlist.takeError()));
  const std::string designStatement = "DESIGN " + top.str() + " ;";
  if (designExchangeFormat.empty() || designExchangeFormat.contains('\0') ||
      designExchangeFormat.contains('\r') ||
      !designExchangeFormat.contains(designStatement) ||
      !designExchangeFormat.contains("\nNETS ") ||
      !designExchangeFormat.contains("\nEND NETS") ||
      !designExchangeFormat.contains("+ ROUTED ") ||
      !designExchangeFormat.contains("END DESIGN"))
    return makeCadenceAdapterError(
        CadenceAdapterFailureKind::ParserFailure,
        descriptor.implementationSemanticIdentity,
        "routed DEF is malformed, lacks signal routing, or names a different "
        "design");
  if (generationConstraints.empty() || generationConstraints.contains('\0') ||
      generationConstraints.contains('\r'))
    return makeCadenceAdapterError(
        CadenceAdapterFailureKind::ParserFailure,
        descriptor.implementationSemanticIdentity,
        "routed constraint snapshot violates the LF text contract");
  return InnovusPhysicalSnapshot{stage, parsedNetlist->verilog,
                                 designExchangeFormat.str(),
                                 generationConstraints.str()};
}

llvm::Expected<external_tool::ExternalToolInvocationBundleSpec>
makeInnovusBundleSpec(const CadenceBundleInputs &inputs, llvm::StringRef top,
                      llvm::StringRef gateNetlist,
                      llvm::StringRef generationConstraint,
                      llvm::StringRef floorplan) {
  const std::vector<std::string> requiredInputs{
      gateNetlist.str(), generationConstraint.str(), floorplan.str()};
  if (llvm::Error error =
          validateCadenceSemanticInputs(descriptor, inputs, requiredInputs))
    return std::move(error);
  const external_tool::ResolvedExternalFile *technologyLef =
      findExternal(inputs, "technology_lef");
  const external_tool::ResolvedExternalFile *cellLef =
      findExternal(inputs, "cell_lef");
  const external_tool::ResolvedExternalFile *timingLiberty =
      findExternal(inputs, "timing_liberty");
  const external_tool::ResolvedExternalFile *qrcTechnology =
      findExternal(inputs, "qrc_technology_file");
  if (!technologyLef || !cellLef || !timingLiberty || !qrcTechnology)
    return makeCadenceAdapterError(
        CadenceAdapterFailureKind::MissingProviderInput,
        descriptor.implementationSemanticIdentity,
        "physical or timing provider input is absent");
  auto driver =
      renderInnovusDriver(top, gateNetlist, floorplan,
                          technologyLef->absolutePath, cellLef->absolutePath);
  if (!driver)
    return driver.takeError();
  auto mmmc =
      renderInnovusMmmcDriver(generationConstraint, timingLiberty->absolutePath,
                              qrcTechnology->absolutePath);
  if (!mmmc)
    return mmmc.takeError();
  std::vector<std::vector<std::string>> commands{{inputs.frozen.tool.executable,
                                                  "-no_gui", "-batch", "-files",
                                                  "drivers/innovus.tcl"}};
  std::vector<external_tool::MaterializedBundleFile> drivers{
      {"drivers/innovus-mmmc.tcl", std::move(*mmmc), std::nullopt, false},
      {"drivers/innovus.tcl", std::move(*driver), std::nullopt, false}};
  return makeCadenceInvocationBundleSpec(
      descriptor, inputs, std::move(commands), std::move(drivers));
}

llvm::Expected<InnovusPhysicalSnapshot> importInnovusPhysicalSnapshot(
    const external_tool::PreparedExternalToolInvocation &prepared,
    const CadenceBundleInputs &inputs, llvm::StringRef top) {
  auto imported = importCadenceInvocation(descriptor, prepared, inputs);
  if (!imported)
    return imported.takeError();
  auto netlist = readCadenceDeclaredOutput(descriptor, *imported,
                                           "outputs/innovus-routed.v");
  if (!netlist)
    return netlist.takeError();
  auto def = readCadenceDeclaredOutput(descriptor, *imported,
                                       "outputs/innovus-routed.def");
  if (!def)
    return def.takeError();
  auto constraints = readCadenceDeclaredOutput(descriptor, *imported,
                                               "outputs/innovus-routed.sdc");
  if (!constraints)
    return constraints.takeError();
  return parseInnovusPhysicalSnapshot(
      *netlist, *def, *constraints, top,
      hardware::RepresentationPhysicalStage::Routed);
}

llvm::Expected<hardware::FinalizedHardwareImplementation>
publishInnovusPhysicalImplementation(
    const hardware::FinalizedHardwareImplementation &source,
    const InnovusPhysicalSnapshot &snapshot, const ArtifactStore &artifacts,
    const BlobStore &blobs) {
  using namespace hardware;
  const HardwareImplementation &input = source.implementation();
  const ImplementationRepresentationRoot &inputRoot =
      input.representationRoot();
  if (inputRoot.variant != RepresentationRootVariant::GateNetlist ||
      inputRoot.formatRef.kind() !=
          RepresentationFormatKind::StructuralVerilogGateNetlist)
    return makeCadenceAdapterError(
        CadenceAdapterFailureKind::UnsupportedImplementation,
        descriptor.implementationSemanticIdentity,
        "physical publication requires the exact finalized GateNetlist");
  if (snapshot.stage != RepresentationPhysicalStage::Routed)
    return makeCadenceAdapterError(
        CadenceAdapterFailureKind::UnsupportedImplementation,
        descriptor.implementationSemanticIdentity,
        "physical publication requires a Routed snapshot");
  if (!input.implementationPlatform())
    return makeCadenceAdapterError(
        CadenceAdapterFailureKind::MissingTarget,
        descriptor.implementationSemanticIdentity,
        "GateNetlist has no exact implementation platform");

  auto checkedSnapshot = parseInnovusPhysicalSnapshot(
      snapshot.netlistVerilog, snapshot.designExchangeFormat,
      snapshot.generationConstraints, inputRoot.top.canonicalName,
      snapshot.stage);
  if (!checkedSnapshot)
    return checkedSnapshot.takeError();
  auto inputIndex = indexRepresentationRoot(inputRoot, blobs);
  if (!inputIndex)
    return makeCadenceAdapterError(
        CadenceAdapterFailureKind::PublicationUnavailable,
        descriptor.implementationSemanticIdentity,
        llvm::toString(inputIndex.takeError()));
  const ImplementationPayloadBytes outputNetlist{
      PayloadRole::Netlist, "netlist/innovus-routed.v",
      llvm::ArrayRef<std::uint8_t>(reinterpret_cast<const std::uint8_t *>(
                                       checkedSnapshot->netlistVerilog.data()),
                                   checkedSnapshot->netlistVerilog.size())};
  auto outputIndex = indexProspectiveRepresentation(
      inputRoot.formatRef, inputRoot.top,
      llvm::ArrayRef<ImplementationPayloadBytes>(&outputNetlist, 1));
  if (!outputIndex)
    return makeCadenceAdapterError(CadenceAdapterFailureKind::ParserFailure,
                                   descriptor.implementationSemanticIdentity,
                                   "routed boundary cannot be indexed: " +
                                       llvm::toString(outputIndex.takeError()));
  if (inputIndex->rootBoundaryPorts() != outputIndex->rootBoundaryPorts() ||
      inputIndex->unresolvedExternalDefinitions() !=
          outputIndex->unresolvedExternalDefinitions())
    return makeCadenceAdapterError(
        CadenceAdapterFailureKind::PublicationUnavailable,
        descriptor.implementationSemanticIdentity,
        "routed netlist changed the exact boundary or external definition "
        "closure");
  auto format = RepresentationFormatDescriptorRef::get(
      RepresentationFormatKind::IndexedDefPhysical);
  if (!format)
    return makeCadenceAdapterError(
        CadenceAdapterFailureKind::PublicationUnavailable,
        descriptor.implementationSemanticIdentity,
        llvm::toString(format.takeError()));

  const auto storeBytes =
      [&](llvm::StringRef contents) -> llvm::Expected<BlobDigest> {
    return blobs.put(llvm::ArrayRef<std::uint8_t>(
        reinterpret_cast<const std::uint8_t *>(contents.data()),
        contents.size()));
  };
  auto netlistDigest = storeBytes(checkedSnapshot->netlistVerilog);
  if (!netlistDigest)
    return netlistDigest.takeError();
  auto databaseDigest = storeBytes(checkedSnapshot->designExchangeFormat);
  if (!databaseDigest)
    return databaseDigest.takeError();
  auto constraintDigest = storeBytes(checkedSnapshot->generationConstraints);
  if (!constraintDigest)
    return constraintDigest.takeError();
  std::vector<ImplementationPayload> payloads{
      {PayloadRole::Netlist, "netlist/innovus-routed.v", *netlistDigest},
      {PayloadRole::PhysicalDatabase, "database/innovus-routed.def",
       *databaseDigest},
      {PayloadRole::GenerationConstraint, "constraints/innovus-routed.sdc",
       *constraintDigest}};
  for (const ImplementationPayload &payload : inputRoot.payloads)
    if (payload.role == PayloadRole::BlackBoxContract)
      payloads.push_back(payload);

  const RepresentationLocator physicalTop{
      RepresentationObjectKind::PhysicalObject, inputRoot.top.canonicalName};
  std::vector<PhysicalRepresentationObject> objects{
      {physicalTop, std::nullopt}};
  const auto addInputObject =
      [&](const RepresentationLocator &locator) -> llvm::Error {
    if (llvm::any_of(objects, [&](const auto &object) {
          return object.locator == locator;
        }))
      return llvm::Error::success();
    auto facts = inputIndex->lookup(locator);
    if (!facts)
      return facts.takeError();
    if (!*facts)
      return makeCadenceAdapterError(
          CadenceAdapterFailureKind::PublicationUnavailable,
          descriptor.implementationSemanticIdentity,
          "source representation does not index locator '" +
              locator.canonicalName + "'");
    auto routedFacts = outputIndex->lookup(locator);
    if (!routedFacts)
      return routedFacts.takeError();
    if (!*routedFacts || !(**routedFacts == **facts))
      return makeCadenceAdapterError(
          CadenceAdapterFailureKind::PublicationUnavailable,
          descriptor.implementationSemanticIdentity,
          "routed netlist changed or omitted locator '" +
              locator.canonicalName + "'");
    objects.push_back({locator, (*facts)->signalGeometry});
    return llvm::Error::success();
  };
  for (const ImplementationInterface &interface : input.interfaces())
    if (llvm::Error error = addInputObject(interface.representationLocator))
      return std::move(error);
  for (const ActivityPoint &activity : input.activityPoints())
    if (llvm::Error error = addInputObject(activity.representationLocator))
      return std::move(error);
  for (const MemoryMacroBinding &memory : input.memoryMacroBindings())
    if (llvm::Error error = addInputObject(memory.representationLocator))
      return std::move(error);
  for (const ExternalImplementationBinding &binding :
       input.externalImplementationBindings())
    for (const RepresentationLocator &locator : binding.representationLocators)
      if (llvm::Error error = addInputObject(locator))
        return std::move(error);

  auto physicalIndex = createPhysicalRepresentationIndexPayload(
      *format, RepresentationRootVariant::AsicPhysical,
      RepresentationPhysicalStage::Routed, physicalTop,
      "index/innovus-physical.json", payloads, std::move(objects),
      outputIndex->unresolvedExternalDefinitions().vec());
  if (!physicalIndex)
    return makeCadenceAdapterError(
        CadenceAdapterFailureKind::PublicationUnavailable,
        descriptor.implementationSemanticIdentity,
        llvm::toString(physicalIndex.takeError()));
  auto indexBytes =
      serializePhysicalRepresentationIndexPayloadJson(*physicalIndex);
  if (!indexBytes)
    return indexBytes.takeError();
  auto indexDigest = storeBytes(*indexBytes);
  if (!indexDigest)
    return indexDigest.takeError();
  payloads.push_back({PayloadRole::RepresentationIndex,
                      "index/innovus-physical.json", *indexDigest});
  auto representation = createImplementationRepresentationRoot(
      RepresentationRootVariant::AsicPhysical,
      RepresentationPhysicalStage::Routed, *format, physicalTop,
      std::move(payloads));
  if (!representation)
    return makeCadenceAdapterError(
        CadenceAdapterFailureKind::PublicationUnavailable,
        descriptor.implementationSemanticIdentity,
        llvm::toString(representation.takeError()));

  std::vector<ExternalImplementationBindingDraft> externalBindings;
  for (const ExternalImplementationBinding &binding :
       input.externalImplementationBindings()) {
    std::optional<ImplementationPayloadKey> blackBox;
    if (binding.blackBoxContractPayloadRef) {
      const std::uint64_t ordinal = binding.blackBoxContractPayloadRef->ordinal;
      if (ordinal >= inputRoot.payloads.size())
        return makeCadenceAdapterError(
            CadenceAdapterFailureKind::PublicationUnavailable,
            descriptor.implementationSemanticIdentity,
            "source external binding has an invalid payload reference");
      const ImplementationPayload &payload = inputRoot.payloads[ordinal];
      blackBox =
          ImplementationPayloadKey{payload.role, payload.canonicalLogicalName};
    }
    externalBindings.push_back(
        {binding.providerContractRef, binding.externalInputs,
         binding.fabricResourceRefs, binding.representationLocators,
         std::move(blackBox)});
  }
  std::vector<MemoryMacroBindingDraft> memoryBindings;
  for (const MemoryMacroBinding &binding : input.memoryMacroBindings())
    memoryBindings.push_back({binding.fabricMemoryRef,
                              binding.externalImplementationBindingRef.ordinal,
                              binding.representationLocator});

  HardwareImplementationDraft draft{input.fabric(),
                                    input.configurationAbi(),
                                    input.interconnectImplementations().vec(),
                                    std::move(*representation),
                                    input.implementationPlatform(),
                                    input.interfaces().vec(),
                                    input.activityPoints().vec(),
                                    std::move(memoryBindings),
                                    std::move(externalBindings)};
  auto catalog = makeCadenceStandardCellContractCatalog();
  if (!catalog)
    return catalog.takeError();
  auto finalized = finalizeHardwareImplementation(std::move(draft), *catalog,
                                                  artifacts, blobs);
  if (!finalized)
    return makeCadenceAdapterError(
        CadenceAdapterFailureKind::PublicationUnavailable,
        descriptor.implementationSemanticIdentity,
        llvm::toString(finalized.takeError()));
  auto strict = importHardwareImplementation(finalized->reference(), *catalog,
                                             artifacts, blobs);
  if (!strict)
    return makeCadenceAdapterError(
        CadenceAdapterFailureKind::PublicationUnavailable,
        descriptor.implementationSemanticIdentity,
        llvm::toString(strict.takeError()));
  return std::move(*strict);
}

} // namespace loom::eda::cadence
