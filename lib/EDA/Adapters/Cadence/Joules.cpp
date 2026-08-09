#include "EDA/Adapters/Cadence/Joules.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CheckedArithmetic.h"

#include <array>
#include <cstdint>
#include <limits>
#include <optional>
#include <set>
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
    "outputs/joules-power-result.csv"};

const CadenceInvocationDescriptor descriptor{
    &external_tool::joulesProvider(),
    "loom.eda.cadence.joules.power@1",
    CadenceOperation::PowerEvaluation,
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

llvm::Error parserError(const llvm::Twine &detail) {
  return makeCadenceAdapterError(CadenceAdapterFailureKind::ParserFailure,
                                 descriptor.implementationSemanticIdentity,
                                 detail);
}

llvm::Expected<evaluation::DecimalValue>
addPositiveDecimals(evaluation::DecimalValue lhs,
                    evaluation::DecimalValue rhs) {
  if (lhs.coefficient() == 0)
    return rhs;
  if (rhs.coefficient() == 0)
    return lhs;
  const std::int64_t exponent =
      std::min(lhs.base10Exponent(), rhs.base10Exponent());
  const auto scaled =
      [&](evaluation::DecimalValue value) -> llvm::Expected<std::uint64_t> {
    const __int128 difference = static_cast<__int128>(value.base10Exponent()) -
                                static_cast<__int128>(exponent);
    if (difference < 0 || difference > 18)
      return parserError("power components cannot be added exactly");
    std::uint64_t scale = 1;
    for (__int128 index = 0; index < difference; ++index)
      scale *= 10;
    auto product = llvm::checkedMulUnsigned(
        static_cast<std::uint64_t>(value.coefficient()), scale);
    if (!product || *product > static_cast<std::uint64_t>(
                                   std::numeric_limits<std::int64_t>::max()))
      return parserError("power component addition overflows");
    return *product;
  };
  auto scaledLhs = scaled(lhs);
  if (!scaledLhs)
    return scaledLhs.takeError();
  auto scaledRhs = scaled(rhs);
  if (!scaledRhs)
    return scaledRhs.takeError();
  auto sum = llvm::checkedAddUnsigned(*scaledLhs, *scaledRhs);
  if (!sum || *sum > static_cast<std::uint64_t>(
                         std::numeric_limits<std::int64_t>::max()))
    return parserError("power component addition overflows");
  return evaluation::DecimalValue::get(static_cast<std::int64_t>(*sum),
                                       exponent);
}

} // namespace

const CadenceInvocationDescriptor &joulesPowerDescriptor() {
  return descriptor;
}

llvm::Expected<std::string>
renderJoulesPowerDriver(llvm::StringRef top, llvm::StringRef gateNetlist,
                        llvm::StringRef generationConstraint,
                        llvm::StringRef activity, llvm::StringRef activityScope,
                        llvm::StringRef timingLiberty) {
  if (!isPortableHdlIdentifier(top))
    return makeCadenceAdapterError(
        CadenceAdapterFailureKind::MissingSemanticInput,
        descriptor.implementationSemanticIdentity,
        "top is not a portable HDL identifier");
  auto topWord = renderTclWord(descriptor.implementationSemanticIdentity, top);
  auto netlistWord = inputWord(gateNetlist);
  auto constraintWord = inputWord(generationConstraint);
  auto activityWord = inputWord(activity);
  auto scopeWord =
      renderTclWord(descriptor.implementationSemanticIdentity, activityScope);
  auto libraryWord =
      renderTclWord(descriptor.implementationSemanticIdentity, timingLiberty);
  if (!topWord)
    return topWord.takeError();
  if (!netlistWord)
    return netlistWord.takeError();
  if (!constraintWord)
    return constraintWord.takeError();
  if (!activityWord)
    return activityWord.takeError();
  if (!scopeWord)
    return scopeWord.takeError();
  if (!libraryWord)
    return libraryWord.takeError();

  const std::string commands = "read_libs " + *libraryWord +
                               "\n"
                               "read_hdl -sv " +
                               *netlistWord +
                               "\n"
                               "elaborate " +
                               *topWord +
                               "\n"
                               "read_sdc " +
                               *constraintWord +
                               "\n"
                               "read_stimulus -file " +
                               *activityWord + " -dut_instance " + *scopeWord +
                               " -format saif"
                               "\n"
                               "compute_power -mode time_based\n";
  return renderCadenceTclBatch(commands,
                               "report_power -unit W -format %.17g -csv -out "
                               "{outputs/joules-power-result.csv}\n");
}

llvm::Expected<JoulesPowerObservation>
parseJoulesPowerObservation(llvm::StringRef contents) {
  llvm::SmallVector<llvm::StringRef, 16> lines;
  contents.split(lines, '\n', -1, true);
  if (!lines.empty() && lines.back().empty())
    lines.pop_back();
  for (llvm::StringRef &line : lines)
    line.consume_back("\r");
  if (lines.size() < 6 || !lines[0].consume_front("Instance: ") ||
      lines[0].empty() || lines[1] != "Power Unit: W" ||
      !lines[2].consume_front("PDB Frames: ") || lines[2].empty() ||
      lines[3] != "Category,leakage,internal,switching,total,Row%")
    return parserError("power report envelope is invalid");

  std::set<std::string> categories;
  std::optional<evaluation::DecimalValue> subtotalLeakage;
  std::optional<evaluation::DecimalValue> subtotalInternal;
  std::optional<evaluation::DecimalValue> subtotalSwitching;
  bool sawPercentages = false;
  const auto parsePercent = [&](llvm::StringRef value) -> llvm::Error {
    if (!value.consume_back("%"))
      return parserError("power report percentage is invalid");
    auto parsed = parseCadenceDecimal(descriptor.implementationSemanticIdentity,
                                      "power percentage", value, true);
    if (!parsed)
      return parsed.takeError();
    return llvm::Error::success();
  };

  for (std::size_t index = 4; index < lines.size(); ++index) {
    llvm::SmallVector<llvm::StringRef, 6> fields;
    lines[index].split(fields, ',', -1, true);
    if (fields.size() != 6 || llvm::any_of(fields, [](llvm::StringRef field) {
          return field.empty();
        }))
      return parserError("power report row is invalid");
    if (!categories.insert(fields[0].str()).second)
      return parserError("power report category is duplicated");
    if (fields[0] == "Percentage") {
      if (index + 1 != lines.size() || !subtotalLeakage)
        return parserError("power report percentage row is misplaced");
      for (llvm::StringRef field : llvm::drop_begin(fields))
        if (llvm::Error error = parsePercent(field))
          return std::move(error);
      sawPercentages = true;
      continue;
    }
    if (sawPercentages || subtotalLeakage)
      return parserError("power report data follows its subtotal");

    std::array<std::optional<evaluation::DecimalValue>, 4> values;
    for (std::size_t field = 1; field != 5; ++field) {
      auto parsed =
          parseCadenceDecimal(descriptor.implementationSemanticIdentity,
                              "power report value", fields[field], true);
      if (!parsed)
        return parsed.takeError();
      values[field - 1] = *parsed;
    }
    if (llvm::Error error = parsePercent(fields[5]))
      return std::move(error);
    if (fields[0] == "Subtotal") {
      subtotalLeakage = *values[0];
      subtotalInternal = *values[1];
      subtotalSwitching = *values[2];
    }
  }
  if (!subtotalLeakage || !subtotalInternal || !subtotalSwitching ||
      !sawPercentages)
    return parserError("power report is incomplete");
  auto dynamic = addPositiveDecimals(*subtotalInternal, *subtotalSwitching);
  if (!dynamic)
    return dynamic.takeError();
  return JoulesPowerObservation{*dynamic, *subtotalLeakage};
}

llvm::Expected<external_tool::ExternalToolInvocationBundleSpec>
makeJoulesPowerBundleSpec(const CadenceBundleInputs &inputs,
                          llvm::StringRef top, llvm::StringRef gateNetlist,
                          llvm::StringRef generationConstraint,
                          llvm::StringRef activity,
                          llvm::StringRef activityScope) {
  const std::vector<std::string> requiredInputs{
      gateNetlist.str(), generationConstraint.str(), activity.str()};
  if (llvm::Error error =
          validateCadenceSemanticInputs(descriptor, inputs, requiredInputs))
    return std::move(error);
  const auto *library = findExternal(inputs, "timing_liberty");
  if (!library)
    return makeCadenceAdapterError(
        CadenceAdapterFailureKind::MissingProviderInput,
        descriptor.implementationSemanticIdentity, "timing_liberty is absent");
  auto driver =
      renderJoulesPowerDriver(top, gateNetlist, generationConstraint, activity,
                              activityScope, library->absolutePath);
  if (!driver)
    return driver.takeError();
  return makeCadenceInvocationBundleSpec(
      descriptor, inputs,
      {{inputs.frozen.tool.executable, "-files", "drivers/joules.tcl"}},
      {{"drivers/joules.tcl", std::move(*driver), std::nullopt, false}});
}

llvm::Expected<JoulesPowerObservation> importJoulesPowerObservation(
    const external_tool::PreparedExternalToolInvocation &prepared,
    const CadenceBundleInputs &inputs) {
  auto imported = importCadenceInvocation(descriptor, prepared, inputs);
  if (!imported)
    return imported.takeError();
  auto contents = readCadenceDeclaredOutput(descriptor, *imported,
                                            descriptor.declaredOutputs.front());
  if (!contents)
    return contents.takeError();
  return parseJoulesPowerObservation(*contents);
}

} // namespace loom::eda::cadence
