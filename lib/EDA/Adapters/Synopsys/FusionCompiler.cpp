#include "EDA/Adapters/Synopsys/FusionCompiler.h"

#include "EDA/Adapters/Synopsys/DesignCompiler.h"

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
  const external_tool::ResolvedExternalFile *library =
      findExternal(inputs, "reference_library");
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

llvm::Error fusionCompilerPublicationUnavailable() {
  return makeSynopsysAdapterError(
      SynopsysAdapterFailureKind::PublicationUnavailable,
      descriptor.implementationSemanticIdentity,
      "the imported snapshot does not contain the required PhysicalDatabase "
      "and provider-produced RepresentationIndex payloads");
}

} // namespace loom::eda::synopsys
