#include "EDA/Adapters/Cadence/Tempus.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/JSON.h"

#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace loom::eda::cadence {
namespace {

constexpr CadenceImplementationState acceptedStates[]{
    {hardware::RepresentationRootVariant::AsicPhysical,
     hardware::RepresentationPhysicalStage::Routed},
};
constexpr llvm::StringLiteral providerInputs[]{"timing_liberty"};
constexpr llvm::StringLiteral declaredOutputs[]{
    "outputs/tempus-timing-result.json"};

const CadenceInvocationDescriptor descriptor{
    "tempus",
    "loom.eda.cadence.tempus.timing@1",
    CadenceOperation::TimingEvaluation,
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

llvm::Expected<std::string> inputWord(llvm::StringRef value) {
  if (llvm::Error error = validateBundleInputPath(
          descriptor.implementationSemanticIdentity, value))
    return std::move(error);
  return renderTclWord(descriptor.implementationSemanticIdentity, value);
}

llvm::Expected<std::string> externalWord(llvm::StringRef value) {
  return renderTclWord(descriptor.implementationSemanticIdentity, value);
}

llvm::Error parserError(const llvm::Twine &detail) {
  return makeCadenceAdapterError(CadenceAdapterFailureKind::ParserFailure,
                                 descriptor.implementationSemanticIdentity,
                                 detail);
}

} // namespace

const CadenceInvocationDescriptor &tempusTimingDescriptor() {
  return descriptor;
}

llvm::Expected<std::string>
renderTempusTimingDriver(llvm::StringRef top, llvm::StringRef gateNetlist,
                         llvm::StringRef generationConstraint,
                         llvm::StringRef physicalDatabase,
                         llvm::StringRef timingLiberty) {
  if (!isPortableHdlIdentifier(top))
    return makeCadenceAdapterError(
        CadenceAdapterFailureKind::MissingSemanticInput,
        descriptor.implementationSemanticIdentity,
        "top is not a portable HDL identifier");
  auto topWord = externalWord(top);
  auto netlistWord = inputWord(gateNetlist);
  auto constraintWord = inputWord(generationConstraint);
  auto physicalWord = inputWord(physicalDatabase);
  auto libertyWord = externalWord(timingLiberty);
  if (!topWord)
    return topWord.takeError();
  if (!netlistWord)
    return netlistWord.takeError();
  if (!constraintWord)
    return constraintWord.takeError();
  if (!physicalWord)
    return physicalWord.takeError();
  if (!libertyWord)
    return libertyWord.takeError();

  const std::string commands =
      "read_libs " + *libertyWord +
      "\n"
      "read_netlist " +
      *netlistWord + " -top " + *topWord +
      "\n"
      "read_sdc " +
      *constraintWord +
      "\n"
      "read_def " +
      *physicalWord +
      "\n"
      "update_timing\n"
      "set loom_clocks [get_db clocks]\n"
      "if {[llength $loom_clocks] != 1} {error {expected one clock}}\n"
      "set loom_paths [get_db timing_paths -if {.path_type == max} -limit 1]\n"
      "if {[llength $loom_paths] != 1} {error {expected one setup path}}\n"
      "set loom_target_period_ns [get_db [lindex $loom_clocks 0] .period]\n"
      "set loom_worst_slack_ns [get_db [lindex $loom_paths 0] .slack]\n"
      "set loom_period_ns [expr {double($loom_target_period_ns) - "
      "double($loom_worst_slack_ns)}]\n"
      "if {$loom_period_ns <= 0.0} {error {limiting period is not positive}}\n"
      "set loom_period_seconds [expr {$loom_period_ns * 1.0e-9}]\n"
      "set loom_frequency_hz [expr {1.0 / $loom_period_seconds}]\n";
  return renderCadenceTclBatch(
      commands, "set loom_output [open {outputs/tempus-timing-result.json} w]\n"
                "puts $loom_output [format "
                "{{\"schema\":\"loom.cadence.tempus_timing_result\","
                "\"version\":\"1.0\",\"clock_period_seconds\":\"%.17g\","
                "\"limiting_clock_frequency_hz\":\"%.17g\"}} "
                "$loom_period_seconds $loom_frequency_hz]\n"
                "close $loom_output\n");
}

llvm::Expected<TempusTimingObservation>
parseTempusTimingObservation(llvm::StringRef contents) {
  auto parsed = llvm::json::parse(contents);
  if (!parsed)
    return parserError("timing result is malformed JSON: " +
                       llvm::toString(parsed.takeError()));
  const llvm::json::Object *object = parsed->getAsObject();
  if (!object || object->size() != 4)
    return parserError("timing result shape is invalid");
  const auto schema = object->getString("schema");
  const auto version = object->getString("version");
  const auto period = object->getString("clock_period_seconds");
  const auto frequency = object->getString("limiting_clock_frequency_hz");
  if (!schema || *schema != "loom.cadence.tempus_timing_result" || !version ||
      *version != "1.0" || !period || !frequency)
    return parserError("timing result fields are invalid");
  auto parsedPeriod =
      parseCadenceDecimal(descriptor.implementationSemanticIdentity,
                          "clock_period_seconds", *period, false);
  if (!parsedPeriod)
    return parsedPeriod.takeError();
  auto parsedFrequency =
      parseCadenceDecimal(descriptor.implementationSemanticIdentity,
                          "limiting_clock_frequency_hz", *frequency, false);
  if (!parsedFrequency)
    return parsedFrequency.takeError();
  return TempusTimingObservation{*parsedPeriod, *parsedFrequency};
}

llvm::Expected<external_tool::ExternalToolInvocationBundleSpec>
makeTempusTimingBundleSpec(const CadenceBundleInputs &inputs,
                           llvm::StringRef top, llvm::StringRef gateNetlist,
                           llvm::StringRef generationConstraint,
                           llvm::StringRef physicalDatabase) {
  const std::vector<std::string> requiredInputs{
      gateNetlist.str(), generationConstraint.str(), physicalDatabase.str()};
  if (llvm::Error error =
          validateCadenceSemanticInputs(descriptor, inputs, requiredInputs))
    return std::move(error);
  const auto *library = findExternal(inputs, "timing_liberty");
  if (!library)
    return makeCadenceAdapterError(
        CadenceAdapterFailureKind::MissingProviderInput,
        descriptor.implementationSemanticIdentity, "timing_liberty is absent");
  auto driver =
      renderTempusTimingDriver(top, gateNetlist, generationConstraint,
                               physicalDatabase, library->absolutePath);
  if (!driver)
    return driver.takeError();
  return makeCadenceInvocationBundleSpec(
      descriptor, inputs,
      {{inputs.frozen.tool.executable, "-no_gui", "-files",
        "drivers/tempus.tcl"}},
      {{"drivers/tempus.tcl", std::move(*driver), std::nullopt, false}});
}

llvm::Expected<TempusTimingObservation> importTempusTimingObservation(
    const external_tool::PreparedExternalToolInvocation &prepared,
    const CadenceBundleInputs &inputs) {
  auto imported = importCadenceInvocation(descriptor, prepared, inputs);
  if (!imported)
    return imported.takeError();
  auto contents = readCadenceDeclaredOutput(descriptor, *imported,
                                            descriptor.declaredOutputs.front());
  if (!contents)
    return contents.takeError();
  return parseTempusTimingObservation(*contents);
}

} // namespace loom::eda::cadence
