#include "EDA/Adapters/Synopsys/PrimeTime.h"

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
constexpr llvm::StringLiteral providerInputs[]{"timing_library"};
constexpr llvm::StringLiteral declaredOutputs[]{
    "outputs/primetime-timing-result.json"};

const SynopsysInvocationDescriptor descriptor{
    &external_tool::primeTimeProvider(),
    "loom.eda.synopsys.primetime.timing@1",
    SynopsysOperation::TimingEvaluation,
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

const SynopsysInvocationDescriptor &primeTimeDescriptor() { return descriptor; }

llvm::Expected<std::string>
renderPrimeTimeDriver(llvm::StringRef top, llvm::StringRef gateNetlist,
                      llvm::StringRef generationConstraint,
                      llvm::StringRef timingLibrary) {
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
  auto libraryWord =
      renderTclWord(descriptor.implementationSemanticIdentity, timingLibrary);
  if (!libraryWord)
    return libraryWord.takeError();

  const std::string commands =
      "set_app_var link_path [concat {*} [list " + *libraryWord +
      "]]\n"
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
      "update_timing\n"
      "set loom_clocks [get_clocks *]\n"
      "if {[sizeof_collection $loom_clocks] != 1} {error {expected one "
      "clock}}\n"
      "set loom_limiting_paths [get_timing_paths -delay_type max "
      "-max_paths 1 -nworst 1]\n"
      "if {[sizeof_collection $loom_limiting_paths] != 1} {error {expected "
      "one limiting setup path}}\n"
      "set loom_limiting_path [index_collection $loom_limiting_paths 0]\n"
      "set loom_target_period_ns [get_attribute $loom_clocks period]\n"
      "set loom_worst_slack_ns [get_attribute $loom_limiting_path slack]\n"
      "set loom_period_ns [expr {double($loom_target_period_ns) - "
      "double($loom_worst_slack_ns)}]\n"
      "if {$loom_period_ns <= 0.0} {error {limiting period is not "
      "positive}}\n"
      "set loom_period_seconds [expr {double($loom_period_ns) * 1.0e-9}]\n"
      "set loom_frequency_hz [expr {1.0 / $loom_period_seconds}]\n";
  return renderSynopsysTclBatch(
      commands,
      "set loom_output [open {outputs/primetime-timing-result.json} w]\n"
      "puts $loom_output [format "
      "{{\"schema\":\"loom.synopsys.primetime_timing_result\","
      "\"version\":\"1.0\",\"clock_period_seconds\":\"%.17g\","
      "\"limiting_clock_frequency_hz\":\"%.17g\"}} "
      "$loom_period_seconds $loom_frequency_hz]\n"
      "close $loom_output\n");
}

llvm::Expected<PrimeTimeObservation>
parsePrimeTimeObservation(llvm::StringRef contents) {
  auto parsed = llvm::json::parse(contents);
  if (!parsed)
    return makeSynopsysAdapterError(SynopsysAdapterFailureKind::ParserFailure,
                                    descriptor.implementationSemanticIdentity,
                                    "timing result is malformed JSON: " +
                                        llvm::toString(parsed.takeError()));
  const llvm::json::Object *object = parsed->getAsObject();
  if (!object || object->size() != 4)
    return makeSynopsysAdapterError(SynopsysAdapterFailureKind::ParserFailure,
                                    descriptor.implementationSemanticIdentity,
                                    "timing result shape is invalid");
  const std::optional<llvm::StringRef> schema = object->getString("schema");
  const std::optional<llvm::StringRef> version = object->getString("version");
  const std::optional<llvm::StringRef> period =
      object->getString("clock_period_seconds");
  const std::optional<llvm::StringRef> frequency =
      object->getString("limiting_clock_frequency_hz");
  if (!schema || *schema != "loom.synopsys.primetime_timing_result" ||
      !version || *version != "1.0" || !period || !frequency)
    return makeSynopsysAdapterError(SynopsysAdapterFailureKind::ParserFailure,
                                    descriptor.implementationSemanticIdentity,
                                    "timing result fields are invalid");
  auto parsedPeriod =
      parseSynopsysDecimal(descriptor.implementationSemanticIdentity,
                           "clock_period_seconds", *period, false);
  if (!parsedPeriod)
    return parsedPeriod.takeError();
  auto parsedFrequency =
      parseSynopsysDecimal(descriptor.implementationSemanticIdentity,
                           "limiting_clock_frequency_hz", *frequency, false);
  if (!parsedFrequency)
    return parsedFrequency.takeError();
  return PrimeTimeObservation{*parsedPeriod, *parsedFrequency};
}

llvm::Expected<external_tool::ExternalToolInvocationBundleSpec>
makePrimeTimeBundleSpec(const SynopsysBundleInputs &inputs, llvm::StringRef top,
                        llvm::StringRef gateNetlist,
                        llvm::StringRef generationConstraint) {
  const std::vector<std::string> requiredInputs{gateNetlist.str(),
                                                generationConstraint.str()};
  if (llvm::Error error =
          validateSynopsysSemanticInputs(descriptor, inputs, requiredInputs))
    return std::move(error);
  const external_tool::ResolvedExternalFile *library =
      findExternal(inputs, "timing_library");
  if (!library)
    return makeSynopsysAdapterError(
        SynopsysAdapterFailureKind::MissingProviderInput,
        descriptor.implementationSemanticIdentity, "timing_library is absent");
  auto driver = renderPrimeTimeDriver(top, gateNetlist, generationConstraint,
                                      library->absolutePath);
  if (!driver)
    return driver.takeError();
  std::vector<std::vector<std::string>> commands{
      {inputs.frozen.tool.executable, "-f", "drivers/primetime.tcl"}};
  std::vector<external_tool::MaterializedBundleFile> drivers{
      {"drivers/primetime.tcl", std::move(*driver), std::nullopt, false}};
  return makeSynopsysInvocationBundleSpec(
      descriptor, inputs, std::move(commands), std::move(drivers));
}

llvm::Expected<PrimeTimeObservation> importPrimeTimeObservation(
    const external_tool::PreparedExternalToolInvocation &prepared,
    const SynopsysBundleInputs &inputs) {
  auto imported = importSynopsysInvocation(descriptor, prepared, inputs);
  if (!imported)
    return imported.takeError();
  auto contents = readSynopsysDeclaredOutput(
      descriptor, *imported, descriptor.declaredOutputs.front());
  if (!contents)
    return contents.takeError();
  return parsePrimeTimeObservation(*contents);
}

} // namespace loom::eda::synopsys
