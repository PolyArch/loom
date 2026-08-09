#include "EDA/Adapters/Synopsys/FusionCompiler.h"

#include "EDA/Adapters/Synopsys/DesignCompiler.h"
#include "Hardware/Implementation/PhysicalRepresentationIndex.h"
#include "Hardware/Implementation/RepresentationIndex.h"

#include "llvm/ADT/STLExtras.h"

#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace loom::eda::synopsys {
namespace {

constexpr SynopsysImplementationState acceptedStates[]{
    {hardware::RepresentationRootVariant::GateNetlist, std::nullopt}};
constexpr llvm::StringLiteral providerInputs[]{
    "reference_library", "early_parasitic_tech", "late_parasitic_tech",
    "parasitic_layer_map"};
constexpr llvm::StringLiteral declaredOutputs[]{
    "outputs/fusion-compiler-routed.v",
    "outputs/fusion-compiler-routed.def",
    "outputs/fusion-compiler-routed.sdc",
};

const SynopsysInvocationDescriptor descriptor{
    "fc_shell",
    "loom.eda.synopsys.fusion_compiler.asic_physical@1",
    SynopsysOperation::PhysicalImplementation,
    acceptedStates,
    true,
    true,
    true,
    providerInputs,
    declaredOutputs,
};

const external_tool::ResolvedExternalFile *
findExternal(const SynopsysBundleInputs &inputs, llvm::StringRef slot) {
  const auto found =
      llvm::find_if(inputs.frozen.externalFiles, [&](const auto &file) {
        return file.providerInputSlot == slot;
      });
  return found == inputs.frozen.externalFiles.end() ? nullptr : &*found;
}

const external_tool::ResolvedExternalFileTree *
findExternalTree(const SynopsysBundleInputs &inputs, llvm::StringRef slot) {
  const auto found =
      llvm::find_if(inputs.frozen.externalFileTrees, [&](const auto &tree) {
        return tree.providerInputSlot == slot;
      });
  return found == inputs.frozen.externalFileTrees.end() ? nullptr : &*found;
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

const SynopsysInvocationDescriptor &fusionCompilerDescriptor() {
  return descriptor;
}

llvm::Expected<std::string> renderFusionCompilerDriver(
    llvm::StringRef top, llvm::StringRef gateNetlist,
    llvm::StringRef generationConstraint, llvm::StringRef floorplan,
    llvm::StringRef referenceLibrary, llvm::StringRef earlyParasiticTech,
    llvm::StringRef lateParasiticTech, llvm::StringRef parasiticLayerMap) {
  if (!isPortableHdlIdentifier(top))
    return makeSynopsysAdapterError(
        SynopsysAdapterFailureKind::MissingSemanticInput,
        descriptor.implementationSemanticIdentity,
        "top is not a portable HDL identifier");
  auto topWord = checkedWord(top, false);
  if (!topWord)
    return topWord.takeError();
  auto netlistWord = checkedWord(gateNetlist, true);
  if (!netlistWord)
    return netlistWord.takeError();
  auto constraintWord = checkedWord(generationConstraint, true);
  if (!constraintWord)
    return constraintWord.takeError();
  auto floorplanWord = checkedWord(floorplan, true);
  if (!floorplanWord)
    return floorplanWord.takeError();
  auto libraryWord = checkedWord(referenceLibrary, false);
  if (!libraryWord)
    return libraryWord.takeError();
  auto earlyParasiticWord = checkedWord(earlyParasiticTech, false);
  if (!earlyParasiticWord)
    return earlyParasiticWord.takeError();
  auto lateParasiticWord = checkedWord(lateParasiticTech, false);
  if (!lateParasiticWord)
    return lateParasiticWord.takeError();
  auto layerMapWord = checkedWord(parasiticLayerMap, false);
  if (!layerMapWord)
    return layerMapWord.takeError();

  const std::string commands =
      "create_lib {fusion.dlib} -ref_libs [list " + *libraryWord +
      "]\n"
      "read_parasitic_tech -tlup " +
      *earlyParasiticWord + " -layermap " + *layerMapWord +
      " -name {loom_early}\n"
      "read_parasitic_tech -tlup " +
      *lateParasiticWord + " -layermap " + *layerMapWord +
      " -name {loom_late}\n"
      "read_verilog -top " +
      *topWord + " " + *netlistWord +
      "\n"
      "current_design " +
      *topWord +
      "\n"
      "read_sdc " +
      *constraintWord +
      "\n"
      "read_def " +
      *floorplanWord +
      "\n"
      "set_parasitic_parameters -early_spec {loom_early} "
      "-late_spec {loom_late}\n"
      "compile_fusion -from initial_map -to final_opto\n"
      "clock_opt\n"
      "route_auto\n"
      "route_opt\n";
  return renderSynopsysTclBatch(
      commands, "write_verilog {outputs/fusion-compiler-routed.v}\n"
                "write_def {outputs/fusion-compiler-routed.def}\n"
                "write_sdc -output {outputs/fusion-compiler-routed.sdc}\n"
                "save_block\n");
}

llvm::Expected<FusionCompilerPhysicalSnapshot>
parseFusionCompilerPhysicalSnapshot(
    llvm::StringRef netlist, llvm::StringRef designExchangeFormat,
    llvm::StringRef generationConstraints, llvm::StringRef top,
    hardware::RepresentationPhysicalStage stage) {
  if (stage != hardware::RepresentationPhysicalStage::Routed)
    return makeSynopsysAdapterError(
        SynopsysAdapterFailureKind::UnsupportedImplementation,
        descriptor.implementationSemanticIdentity,
        "Fusion Compiler route driver publishes only Routed state");
  auto parsedNetlist = parseDesignCompilerGateNetlist(netlist, top);
  if (!parsedNetlist)
    return makeSynopsysAdapterError(SynopsysAdapterFailureKind::ParserFailure,
                                    descriptor.implementationSemanticIdentity,
                                    llvm::toString(parsedNetlist.takeError()));
  const std::string designStatement = "DESIGN " + top.str() + " ;";
  if (designExchangeFormat.empty() || designExchangeFormat.contains('\0') ||
      designExchangeFormat.contains('\r') ||
      !designExchangeFormat.contains(designStatement) ||
      !designExchangeFormat.contains("END DESIGN"))
    return makeSynopsysAdapterError(
        SynopsysAdapterFailureKind::ParserFailure,
        descriptor.implementationSemanticIdentity,
        "routed DEF is empty, malformed, or names a different design");
  if (generationConstraints.empty() || generationConstraints.contains('\0') ||
      generationConstraints.contains('\r'))
    return makeSynopsysAdapterError(
        SynopsysAdapterFailureKind::ParserFailure,
        descriptor.implementationSemanticIdentity,
        "routed constraint snapshot violates the LF text contract");
  return FusionCompilerPhysicalSnapshot{stage, parsedNetlist->verilog,
                                        designExchangeFormat.str(),
                                        generationConstraints.str()};
}

llvm::Expected<external_tool::ExternalToolInvocationBundleSpec>
makeFusionCompilerBundleSpec(const SynopsysBundleInputs &inputs,
                             llvm::StringRef top, llvm::StringRef gateNetlist,
                             llvm::StringRef generationConstraint,
                             llvm::StringRef floorplan) {
  const std::vector<std::string> requiredInputs{
      gateNetlist.str(), generationConstraint.str(), floorplan.str()};
  if (llvm::Error error =
          validateSynopsysSemanticInputs(descriptor, inputs, requiredInputs))
    return std::move(error);
  const external_tool::ResolvedExternalFileTree *library =
      findExternalTree(inputs, "reference_library");
  const external_tool::ResolvedExternalFile *earlyParasitic =
      findExternal(inputs, "early_parasitic_tech");
  const external_tool::ResolvedExternalFile *lateParasitic =
      findExternal(inputs, "late_parasitic_tech");
  const external_tool::ResolvedExternalFile *layerMap =
      findExternal(inputs, "parasitic_layer_map");
  if (!library || !earlyParasitic || !lateParasitic || !layerMap)
    return makeSynopsysAdapterError(
        SynopsysAdapterFailureKind::MissingProviderInput,
        descriptor.implementationSemanticIdentity,
        "physical reference or parasitic provider input is absent");
  auto driver = renderFusionCompilerDriver(
      top, gateNetlist, generationConstraint, floorplan, library->absolutePath,
      earlyParasitic->absolutePath, lateParasitic->absolutePath,
      layerMap->absolutePath);
  if (!driver)
    return driver.takeError();
  std::vector<std::vector<std::string>> commands{
      {inputs.frozen.tool.executable, "-f", "drivers/fusion-compiler.tcl"}};
  std::vector<external_tool::MaterializedBundleFile> drivers{
      {"drivers/fusion-compiler.tcl", std::move(*driver), std::nullopt, false}};
  return makeSynopsysInvocationBundleSpec(
      descriptor, inputs, std::move(commands), std::move(drivers));
}

llvm::Expected<FusionCompilerPhysicalSnapshot>
importFusionCompilerPhysicalSnapshot(
    const external_tool::PreparedExternalToolInvocation &prepared,
    const SynopsysBundleInputs &inputs, llvm::StringRef top) {
  auto imported = importSynopsysInvocation(descriptor, prepared, inputs);
  if (!imported)
    return imported.takeError();
  auto netlist = readSynopsysDeclaredOutput(descriptor, *imported,
                                            "outputs/fusion-compiler-routed.v");
  if (!netlist)
    return netlist.takeError();
  auto def = readSynopsysDeclaredOutput(descriptor, *imported,
                                        "outputs/fusion-compiler-routed.def");
  if (!def)
    return def.takeError();
  auto constraints = readSynopsysDeclaredOutput(
      descriptor, *imported, "outputs/fusion-compiler-routed.sdc");
  if (!constraints)
    return constraints.takeError();
  return parseFusionCompilerPhysicalSnapshot(
      *netlist, *def, *constraints, top,
      hardware::RepresentationPhysicalStage::Routed);
}

llvm::Expected<hardware::FinalizedHardwareImplementation>
publishFusionCompilerPhysicalImplementation(
    const hardware::FinalizedHardwareImplementation &source,
    const FusionCompilerPhysicalSnapshot &snapshot,
    const ArtifactStore &artifacts, const BlobStore &blobs) {
  using namespace hardware;
  const HardwareImplementation &input = source.implementation();
  const ImplementationRepresentationRoot &inputRoot =
      input.representationRoot();
  if (inputRoot.variant != RepresentationRootVariant::GateNetlist ||
      inputRoot.formatRef.kind() !=
          RepresentationFormatKind::StructuralVerilogGateNetlist)
    return makeSynopsysAdapterError(
        SynopsysAdapterFailureKind::UnsupportedImplementation,
        descriptor.implementationSemanticIdentity,
        "physical publication requires the exact finalized GateNetlist");
  if (!input.implementationPlatform())
    return makeSynopsysAdapterError(
        SynopsysAdapterFailureKind::MissingTarget,
        descriptor.implementationSemanticIdentity,
        "GateNetlist has no exact implementation platform");

  auto checkedSnapshot = parseFusionCompilerPhysicalSnapshot(
      snapshot.netlistVerilog, snapshot.designExchangeFormat,
      snapshot.generationConstraints, inputRoot.top.canonicalName,
      snapshot.stage);
  if (!checkedSnapshot)
    return checkedSnapshot.takeError();
  auto inputIndex = indexRepresentationRoot(inputRoot, blobs);
  if (!inputIndex)
    return makeSynopsysAdapterError(
        SynopsysAdapterFailureKind::PublicationUnavailable,
        descriptor.implementationSemanticIdentity,
        llvm::toString(inputIndex.takeError()));
  auto format = RepresentationFormatDescriptorRef::get(
      RepresentationFormatKind::IndexedPhysical);
  if (!format)
    return makeSynopsysAdapterError(
        SynopsysAdapterFailureKind::PublicationUnavailable,
        descriptor.implementationSemanticIdentity,
        llvm::toString(format.takeError()));

  const RepresentationLocator physicalTop{
      RepresentationObjectKind::PhysicalObject, inputRoot.top.canonicalName};
  const auto projectLocator = [&](const RepresentationLocator &locator) {
    return locator == inputRoot.top ? physicalTop : locator;
  };
  std::vector<PhysicalRepresentationObject> objects{
      {physicalTop, std::nullopt}};
  std::vector<RepresentationLocator> unresolved;
  const auto addReferencedObject =
      [&](const RepresentationLocator &sourceLocator) -> llvm::Error {
    const RepresentationLocator locator = projectLocator(sourceLocator);
    if (llvm::any_of(objects, [&](const auto &object) {
          return object.locator == locator;
        }))
      return llvm::Error::success();
    auto facts = inputIndex->lookup(sourceLocator);
    if (!facts)
      return facts.takeError();
    if (!*facts)
      return makeSynopsysAdapterError(
          SynopsysAdapterFailureKind::PublicationUnavailable,
          descriptor.implementationSemanticIdentity,
          "source representation does not index locator '" +
              sourceLocator.canonicalName + "'");
    objects.push_back({locator, (*facts)->signalGeometry});
    if (locator.kind == RepresentationObjectKind::Module)
      unresolved.push_back(locator);
    return llvm::Error::success();
  };

  std::vector<ImplementationInterface> interfaces(input.interfaces().begin(),
                                                  input.interfaces().end());
  for (ImplementationInterface &interface : interfaces) {
    if (llvm::Error error =
            addReferencedObject(interface.representationLocator))
      return std::move(error);
    interface.representationLocator =
        projectLocator(interface.representationLocator);
  }
  std::vector<ActivityPoint> activityPoints(input.activityPoints().begin(),
                                            input.activityPoints().end());
  for (ActivityPoint &point : activityPoints) {
    if (llvm::Error error = addReferencedObject(point.representationLocator))
      return std::move(error);
    point.representationLocator = projectLocator(point.representationLocator);
  }

  std::vector<ExternalImplementationBindingDraft> externalBindings;
  for (const ExternalImplementationBinding &binding :
       input.externalImplementationBindings()) {
    std::vector<RepresentationLocator> locators(
        binding.representationLocators.begin(),
        binding.representationLocators.end());
    for (RepresentationLocator &locator : locators) {
      if (llvm::Error error = addReferencedObject(locator))
        return std::move(error);
      locator = projectLocator(locator);
    }
    std::optional<ImplementationPayloadKey> blackBoxContract;
    if (binding.blackBoxContractPayloadRef) {
      if (binding.blackBoxContractPayloadRef->ordinal >=
          inputRoot.payloads.size())
        return makeSynopsysAdapterError(
            SynopsysAdapterFailureKind::PublicationUnavailable,
            descriptor.implementationSemanticIdentity,
            "source black-box payload reference is out of range");
      const ImplementationPayload &payload =
          inputRoot.payloads[binding.blackBoxContractPayloadRef->ordinal];
      blackBoxContract =
          ImplementationPayloadKey{payload.role, payload.canonicalLogicalName};
    }
    externalBindings.push_back(ExternalImplementationBindingDraft{
        binding.providerContractRef, binding.externalInputs,
        binding.fabricResourceRefs, std::move(locators),
        std::move(blackBoxContract)});
  }

  std::vector<MemoryMacroBindingDraft> memoryBindings;
  for (const MemoryMacroBinding &binding : input.memoryMacroBindings()) {
    if (binding.externalImplementationBindingRef.ordinal >=
        externalBindings.size())
      return makeSynopsysAdapterError(
          SynopsysAdapterFailureKind::PublicationUnavailable,
          descriptor.implementationSemanticIdentity,
          "source memory binding references an unknown external "
          "implementation");
    if (llvm::Error error = addReferencedObject(binding.representationLocator))
      return std::move(error);
    memoryBindings.push_back(MemoryMacroBindingDraft{
        binding.fabricMemoryRef,
        binding.externalImplementationBindingRef.ordinal,
        projectLocator(binding.representationLocator)});
  }
  for (const RepresentationLocator &locator :
       inputIndex->unresolvedExternalDefinitions())
    if (llvm::Error error = addReferencedObject(locator))
      return std::move(error);

  const auto storeBytes =
      [&](llvm::StringRef contents) -> llvm::Expected<BlobDigest> {
    return blobs.put(llvm::ArrayRef<std::uint8_t>(
        reinterpret_cast<const std::uint8_t *>(contents.data()),
        contents.size()));
  };
  auto databaseDigest = storeBytes(checkedSnapshot->designExchangeFormat);
  if (!databaseDigest)
    return databaseDigest.takeError();
  auto constraintDigest = storeBytes(checkedSnapshot->generationConstraints);
  if (!constraintDigest)
    return constraintDigest.takeError();
  std::vector<ImplementationPayload> payloads{
      {PayloadRole::PhysicalDatabase, "database/fusion-compiler-routed.def",
       *databaseDigest},
      {PayloadRole::GenerationConstraint,
       "constraints/fusion-compiler-routed.sdc", *constraintDigest}};
  for (const ImplementationPayload &payload : inputRoot.payloads)
    if (payload.role == PayloadRole::BlackBoxContract)
      payloads.push_back(payload);

  auto physicalIndex = createPhysicalRepresentationIndexPayload(
      *format, RepresentationRootVariant::AsicPhysical,
      RepresentationPhysicalStage::Routed, physicalTop,
      "index/fusion-compiler-physical.json", payloads, std::move(objects),
      std::move(unresolved));
  if (!physicalIndex)
    return makeSynopsysAdapterError(
        SynopsysAdapterFailureKind::PublicationUnavailable,
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
                      physicalIndex->indexLogicalName, *indexDigest});
  auto representation = createImplementationRepresentationRoot(
      RepresentationRootVariant::AsicPhysical,
      RepresentationPhysicalStage::Routed, *format, physicalTop,
      std::move(payloads));
  if (!representation)
    return makeSynopsysAdapterError(
        SynopsysAdapterFailureKind::PublicationUnavailable,
        descriptor.implementationSemanticIdentity,
        llvm::toString(representation.takeError()));

  auto contracts = makeSynopsysStandardCellContractCatalog();
  if (!contracts)
    return contracts.takeError();
  auto finalized = finalizeHardwareImplementation(
      HardwareImplementationDraft{
          input.fabric(), input.configurationAbi(),
          input.interconnectImplementations().vec(), std::move(*representation),
          input.implementationPlatform(), std::move(interfaces),
          std::move(activityPoints), std::move(memoryBindings),
          std::move(externalBindings)},
      *contracts, artifacts, blobs);
  if (!finalized)
    return makeSynopsysAdapterError(
        SynopsysAdapterFailureKind::PublicationUnavailable,
        descriptor.implementationSemanticIdentity,
        llvm::toString(finalized.takeError()));
  auto strict = importHardwareImplementation(finalized->reference(), *contracts,
                                             artifacts, blobs);
  if (!strict)
    return makeSynopsysAdapterError(
        SynopsysAdapterFailureKind::PublicationUnavailable,
        descriptor.implementationSemanticIdentity,
        llvm::toString(strict.takeError()));
  return std::move(*strict);
}

} // namespace loom::eda::synopsys
