#include "EDA/Adapters/Synopsys/PrimePower.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/JSON.h"

#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace loom::eda::synopsys {
namespace {

constexpr SynopsysImplementationState acceptedStates[]{
    {hardware::RepresentationRootVariant::GateNetlist, std::nullopt},
};
constexpr llvm::StringLiteral providerInputs[]{"power_library"};
constexpr llvm::StringLiteral declaredOutputs[]{
    "outputs/primepower-power-result.json"};

const SynopsysInvocationDescriptor descriptor{
    &external_tool::primeTimeProvider(),
    "loom.eda.synopsys.primepower.power@1",
    SynopsysOperation::PowerEvaluation,
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

llvm::Expected<std::string> checkedInputWord(llvm::StringRef value) {
  if (llvm::Error error = validateBundleInputPath(
          descriptor.implementationSemanticIdentity, value))
    return std::move(error);
  return renderTclWord(descriptor.implementationSemanticIdentity, value);
}

} // namespace

const SynopsysInvocationDescriptor &primePowerDescriptor() {
  return descriptor;
}

llvm::Expected<std::string> renderPrimePowerDriver(
    llvm::StringRef top, llvm::StringRef gateNetlist,
    llvm::StringRef generationConstraint, llvm::StringRef activity,
    llvm::StringRef activityStripPath, llvm::StringRef powerLibrary) {
  if (!isPortableHdlIdentifier(top))
    return makeSynopsysAdapterError(
        SynopsysAdapterFailureKind::MissingSemanticInput,
        descriptor.implementationSemanticIdentity,
        "top is not a portable HDL identifier");
  auto topWord = renderTclWord(descriptor.implementationSemanticIdentity, top);
  if (!topWord)
    return topWord.takeError();
  auto netlistWord = checkedInputWord(gateNetlist);
  if (!netlistWord)
    return netlistWord.takeError();
  auto constraintWord = checkedInputWord(generationConstraint);
  if (!constraintWord)
    return constraintWord.takeError();
  auto activityWord = checkedInputWord(activity);
  if (!activityWord)
    return activityWord.takeError();
  auto activityStripPathWord = renderTclWord(
      descriptor.implementationSemanticIdentity, activityStripPath);
  if (!activityStripPathWord)
    return activityStripPathWord.takeError();
  auto libraryWord =
      renderTclWord(descriptor.implementationSemanticIdentity, powerLibrary);
  if (!libraryWord)
    return libraryWord.takeError();

  const std::string commands =
      "set_app_var link_path [concat {*} [list " + *libraryWord +
      "]]\n"
      "set_app_var power_enable_analysis true\n"
      "read_verilog " +
      *netlistWord +
      "\n"
      "current_design " +
      *topWord +
      "\n"
      "link_design " +
      *topWord +
      "\n"
      "read_sdc " +
      *constraintWord +
      "\n"
      "if {![read_saif " +
      *activityWord + " -strip_path " + *activityStripPathWord +
      "]} {error {SAIF annotated no design objects}}\n"
      "update_power\n"
      "set loom_dynamic_power [get_attribute [current_design] "
      "dynamic_power]\n"
      "set loom_leakage_power [get_attribute [current_design] "
      "leakage_power]\n";
  return renderSynopsysTclBatch(
      commands,
      "set loom_output [open {outputs/primepower-power-result.json} w]\n"
      "puts $loom_output [format "
      "{{\"schema\":\"loom.synopsys.primepower_power_result\","
      "\"version\":\"1.0\",\"dynamic_power_watts\":\"%.17g\","
      "\"leakage_power_watts\":\"%.17g\"}} "
      "$loom_dynamic_power $loom_leakage_power]\n"
      "close $loom_output\n");
}

llvm::Expected<PrimePowerObservation>
parsePrimePowerObservation(llvm::StringRef contents) {
  auto parsed = llvm::json::parse(contents);
  if (!parsed)
    return makeSynopsysAdapterError(SynopsysAdapterFailureKind::ParserFailure,
                                    descriptor.implementationSemanticIdentity,
                                    "power result is malformed JSON: " +
                                        llvm::toString(parsed.takeError()));
  const llvm::json::Object *object = parsed->getAsObject();
  if (!object || object->size() != 4)
    return makeSynopsysAdapterError(SynopsysAdapterFailureKind::ParserFailure,
                                    descriptor.implementationSemanticIdentity,
                                    "power result shape is invalid");
  const std::optional<llvm::StringRef> schema = object->getString("schema");
  const std::optional<llvm::StringRef> version = object->getString("version");
  const std::optional<llvm::StringRef> dynamic =
      object->getString("dynamic_power_watts");
  const std::optional<llvm::StringRef> leakage =
      object->getString("leakage_power_watts");
  if (!schema || *schema != "loom.synopsys.primepower_power_result" ||
      !version || *version != "1.0" || !dynamic || !leakage)
    return makeSynopsysAdapterError(SynopsysAdapterFailureKind::ParserFailure,
                                    descriptor.implementationSemanticIdentity,
                                    "power result fields are invalid");
  auto parsedDynamic =
      parseSynopsysDecimal(descriptor.implementationSemanticIdentity,
                           "dynamic_power_watts", *dynamic, true);
  if (!parsedDynamic)
    return parsedDynamic.takeError();
  auto parsedLeakage =
      parseSynopsysDecimal(descriptor.implementationSemanticIdentity,
                           "leakage_power_watts", *leakage, true);
  if (!parsedLeakage)
    return parsedLeakage.takeError();
  return PrimePowerObservation{*parsedDynamic, *parsedLeakage};
}

llvm::Expected<external_tool::ExternalToolInvocationBundleSpec>
makePrimePowerBundleSpec(const SynopsysBundleInputs &inputs,
                         llvm::StringRef top, llvm::StringRef gateNetlist,
                         llvm::StringRef generationConstraint,
                         llvm::StringRef activity,
                         llvm::StringRef activityStripPath) {
  const std::vector<std::string> requiredInputs{
      gateNetlist.str(), generationConstraint.str(), activity.str()};
  if (llvm::Error error =
          validateSynopsysSemanticInputs(descriptor, inputs, requiredInputs))
    return std::move(error);
  const external_tool::ResolvedExternalFile *library =
      findExternal(inputs, "power_library");
  if (!library)
    return makeSynopsysAdapterError(
        SynopsysAdapterFailureKind::MissingProviderInput,
        descriptor.implementationSemanticIdentity, "power_library is absent");
  auto driver =
      renderPrimePowerDriver(top, gateNetlist, generationConstraint, activity,
                             activityStripPath, library->absolutePath);
  if (!driver)
    return driver.takeError();
  std::vector<std::vector<std::string>> commands{
      {inputs.frozen.tool.executable, "-f", "drivers/primepower.tcl"}};
  std::vector<external_tool::MaterializedBundleFile> drivers{
      {"drivers/primepower.tcl", std::move(*driver), std::nullopt, false}};
  return makeSynopsysInvocationBundleSpec(
      descriptor, inputs, std::move(commands), std::move(drivers));
}

llvm::Expected<PrimePowerObservation> importPrimePowerObservation(
    const external_tool::PreparedExternalToolInvocation &prepared,
    const SynopsysBundleInputs &inputs) {
  auto imported = importSynopsysInvocation(descriptor, prepared, inputs);
  if (!imported)
    return imported.takeError();
  auto contents = readSynopsysDeclaredOutput(
      descriptor, *imported, descriptor.declaredOutputs.front());
  if (!contents)
    return contents.takeError();
  return parsePrimePowerObservation(*contents);
}

} // namespace loom::eda::synopsys
