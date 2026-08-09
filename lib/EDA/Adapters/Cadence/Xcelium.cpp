#include "EDA/Adapters/Cadence/Xcelium.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/JSON.h"

#include <filesystem>
#include <optional>
#include <string>
#include <utility>

namespace loom::eda::cadence {
namespace {

constexpr CadenceImplementationState acceptedStates[]{
    {hardware::RepresentationRootVariant::Rtl, std::nullopt},
};
constexpr llvm::StringLiteral declaredOutputs[]{
    "outputs/xcelium-functional-result.json"};

const CadenceInvocationDescriptor descriptor{
    "xrun",
    "loom.eda.cadence.xcelium.functional@1",
    CadenceOperation::FunctionalEvaluation,
    acceptedStates,
    false,
    false,
    false,
    {},
    declaredOutputs,
};

llvm::Error invalid(const llvm::Twine &detail) {
  return makeCadenceAdapterError(CadenceAdapterFailureKind::ParserFailure,
                                 descriptor.implementationSemanticIdentity,
                                 detail);
}

std::optional<std::uint64_t> getUnsigned(const llvm::json::Object &object,
                                         llvm::StringRef key) {
  const llvm::json::Value *value = object.get(key);
  return value ? value->getAsUINT64() : std::nullopt;
}

std::string serializeResult(const XceliumFunctionalResult &result) {
  std::string text = "{\"schema\":\"loom.cadence.xcelium_functional_result\","
                     "\"version\":\"1.0\",\"status\":\"";
  text +=
      result.status == XceliumFunctionalStatus::Passed ? "passed" : "failed";
  text += "\",\"completed_transactions\":" +
          std::to_string(result.completedTransactions);
  if (result.firstFailingTransaction)
    text += ",\"first_failing_transaction\":" +
            std::to_string(*result.firstFailingTransaction);
  text += "}\n";
  return text;
}

} // namespace

const CadenceInvocationDescriptor &xceliumFunctionalDescriptor() {
  return descriptor;
}

llvm::Expected<std::vector<std::string>>
renderXceliumFunctionalCommand(llvm::StringRef executable,
                               llvm::StringRef testbenchTop,
                               llvm::ArrayRef<std::string> sourcePaths) {
  const std::filesystem::path executablePath(executable.str());
  if (executable.empty() || executable.contains('\0') ||
      !executablePath.is_absolute() ||
      executablePath.lexically_normal() != executablePath)
    return makeCadenceAdapterError(
        CadenceAdapterFailureKind::ExecutableUnavailable,
        descriptor.implementationSemanticIdentity,
        "frozen Xcelium executable is not an absolute normalized path");
  if (!isPortableHdlIdentifier(testbenchTop))
    return makeCadenceAdapterError(
        CadenceAdapterFailureKind::MissingSemanticInput,
        descriptor.implementationSemanticIdentity,
        "testbench top is not a portable HDL identifier");
  if (sourcePaths.empty())
    return makeCadenceAdapterError(
        CadenceAdapterFailureKind::MissingSemanticInput,
        descriptor.implementationSemanticIdentity, "source inventory is empty");
  for (llvm::StringRef source : sourcePaths)
    if (llvm::Error error = validateBundleInputPath(
            descriptor.implementationSemanticIdentity, source))
      return std::move(error);

  std::vector<std::string> command{executable.str(), "-64bit", "-sv", "-top",
                                   testbenchTop.str()};
  command.insert(command.end(), sourcePaths.begin(), sourcePaths.end());
  return command;
}

llvm::Expected<XceliumFunctionalResult>
parseXceliumFunctionalResult(llvm::StringRef contents) {
  auto parsed = llvm::json::parse(contents);
  if (!parsed)
    return invalid("functional result is malformed JSON: " +
                   llvm::toString(parsed.takeError()));
  const llvm::json::Object *object = parsed->getAsObject();
  if (!object)
    return invalid("functional result root is not an object");
  const std::optional<llvm::StringRef> schema = object->getString("schema");
  const std::optional<llvm::StringRef> version = object->getString("version");
  const std::optional<llvm::StringRef> status = object->getString("status");
  const std::optional<std::uint64_t> completed =
      getUnsigned(*object, "completed_transactions");
  if (!schema || *schema != "loom.cadence.xcelium_functional_result" ||
      !version || *version != "1.0" || !status || !completed || *completed == 0)
    return invalid("functional result fields are invalid");

  XceliumFunctionalResult result;
  result.completedTransactions = *completed;
  const std::optional<std::uint64_t> firstFailing =
      getUnsigned(*object, "first_failing_transaction");
  if (*status == "passed") {
    if (object->size() != 4 || firstFailing)
      return invalid("passed functional result has inconsistent fields");
    result.status = XceliumFunctionalStatus::Passed;
  } else if (*status == "failed") {
    if (object->size() != 5 || !firstFailing || *firstFailing >= *completed)
      return invalid("failed functional result has inconsistent fields");
    result.status = XceliumFunctionalStatus::Failed;
    result.firstFailingTransaction = *firstFailing;
  } else {
    return invalid("functional result status is unknown");
  }
  if (contents != serializeResult(result))
    return invalid("functional result is not canonically encoded");
  return result;
}

llvm::Expected<external_tool::ExternalToolInvocationBundleSpec>
makeXceliumFunctionalBundleSpec(const CadenceBundleInputs &inputs,
                                llvm::StringRef testbenchTop,
                                llvm::ArrayRef<std::string> sourcePaths) {
  if (llvm::Error error =
          validateCadenceSemanticInputs(descriptor, inputs, sourcePaths))
    return std::move(error);
  auto command = renderXceliumFunctionalCommand(inputs.frozen.tool.executable,
                                                testbenchTop, sourcePaths);
  if (!command)
    return command.takeError();
  return makeCadenceInvocationBundleSpec(descriptor, inputs,
                                         {std::move(*command)}, {});
}

llvm::Expected<XceliumFunctionalResult> importXceliumFunctionalResult(
    const external_tool::PreparedExternalToolInvocation &prepared,
    const CadenceBundleInputs &inputs) {
  auto imported = importCadenceInvocation(descriptor, prepared, inputs);
  if (!imported)
    return imported.takeError();
  auto contents = readCadenceDeclaredOutput(descriptor, *imported,
                                            descriptor.declaredOutputs.front());
  if (!contents)
    return contents.takeError();
  return parseXceliumFunctionalResult(*contents);
}

} // namespace loom::eda::cadence
