#include "EDA/Adapters/Synopsys/Common.h"

#include "Common/BlobDigest.h"
#include "ImplementationPlatform/TechnologyCorner.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"

#include <algorithm>
#include <charconv>
#include <filesystem>
#include <limits>
#include <set>
#include <string>
#include <utility>

namespace loom::eda::synopsys {
namespace {

llvm::Error mapIntegrityError(const SynopsysInvocationDescriptor &descriptor,
                              llvm::Error error) {
  return makeSynopsysAdapterError(SynopsysAdapterFailureKind::IntegrityFailure,
                                  descriptor.implementationSemanticIdentity,
                                  llvm::toString(std::move(error)));
}

llvm::Error validateTarget(const SynopsysInvocationDescriptor &descriptor,
                           const SynopsysBundleInputs &inputs) {
  if (!descriptor.requiresAsicPlatform && !descriptor.requiresTechnologyCorner)
    return llvm::Error::success();
  if (!inputs.implementationPlatform || !inputs.platform)
    return makeSynopsysAdapterError(SynopsysAdapterFailureKind::MissingTarget,
                                    descriptor.implementationSemanticIdentity,
                                    "exact ASIC platform is absent");
  if (*inputs.implementationPlatform != inputs.platform->reference())
    return makeSynopsysAdapterError(
        SynopsysAdapterFailureKind::DescriptorMismatch,
        descriptor.implementationSemanticIdentity,
        "implementation and invocation platform identities differ");
  if (!std::holds_alternative<platform::AsicTarget>(
          inputs.platform->platform().target()))
    return makeSynopsysAdapterError(
        SynopsysAdapterFailureKind::MissingTarget,
        descriptor.implementationSemanticIdentity,
        "bound implementation platform is not ASIC");
  if (!descriptor.requiresTechnologyCorner)
    return llvm::Error::success();
  if (!inputs.technologyCorner)
    return makeSynopsysAdapterError(SynopsysAdapterFailureKind::MissingCorner,
                                    descriptor.implementationSemanticIdentity,
                                    "exact technology corner is absent");
  auto corner = platform::decodeTechnologyCornerRef(*inputs.technologyCorner);
  if (!corner)
    return makeSynopsysAdapterError(SynopsysAdapterFailureKind::MissingCorner,
                                    descriptor.implementationSemanticIdentity,
                                    llvm::toString(corner.takeError()));
  if (corner->artifact != inputs.platform->reference().artifact ||
      !inputs.platform->platform().findTechnologyCorner(corner->entity))
    return makeSynopsysAdapterError(
        SynopsysAdapterFailureKind::MissingCorner,
        descriptor.implementationSemanticIdentity,
        "technology corner does not resolve in the exact platform");
  return llvm::Error::success();
}

llvm::Error validateBundleInputs(const SynopsysInvocationDescriptor &descriptor,
                                 const SynopsysBundleInputs &inputs) {
  const bool generatorOperation =
      descriptor.operation == SynopsysOperation::LogicSynthesis ||
      descriptor.operation == SynopsysOperation::PhysicalImplementation;
  const bool generatorClosure = std::holds_alternative<
      external_tool::CandidateGeneratorInvocationClosure>(
      inputs.semanticContract.semanticClosure);
  if (inputs.semanticContract.providerIdentity !=
      descriptor.implementationSemanticIdentity)
    return makeSynopsysAdapterError(
        SynopsysAdapterFailureKind::DescriptorMismatch,
        descriptor.implementationSemanticIdentity,
        "semantic contract provider does not match the adapter");
  if (generatorOperation != generatorClosure)
    return makeSynopsysAdapterError(
        SynopsysAdapterFailureKind::DescriptorMismatch,
        descriptor.implementationSemanticIdentity,
        "semantic invocation closure does not match the adapter operation");
  if (!inputs.implementation)
    return makeSynopsysAdapterError(
        SynopsysAdapterFailureKind::MissingSemanticInput,
        descriptor.implementationSemanticIdentity,
        "exact HardwareImplementation representation is absent");
  if (llvm::Error error =
          validateSynopsysRepresentation(descriptor, *inputs.implementation))
    return error;
  if (llvm::Error error = validateTarget(descriptor, inputs))
    return error;
  if (!descriptor.toolProvider)
    return makeSynopsysAdapterError(
        SynopsysAdapterFailureKind::DescriptorMismatch,
        descriptor.implementationSemanticIdentity,
        "adapter has no backend tool catalog entry");
  const auto *catalogEntry =
      external_tool::findBackendTool(descriptor.toolProvider->binding.key);
  if (!catalogEntry || &catalogEntry->provider != descriptor.toolProvider)
    return makeSynopsysAdapterError(
        SynopsysAdapterFailureKind::DescriptorMismatch,
        descriptor.implementationSemanticIdentity,
        "adapter provider is not owned by the backend tool catalog");
  if (inputs.frozen.tool.toolKey != descriptor.toolProvider->binding.key)
    return makeSynopsysAdapterError(
        SynopsysAdapterFailureKind::DescriptorMismatch,
        descriptor.implementationSemanticIdentity,
        "frozen tool binding has key '" + inputs.frozen.tool.toolKey + "'");
  if (llvm::Error error = validateSynopsysProviderInputs(
          descriptor, inputs.frozen.externalFiles,
          inputs.frozen.externalFileTrees))
    return error;
  for (const external_tool::MaterializedBundleFile &file :
       inputs.semanticInputs) {
    if (!file.sourceArtifact)
      return makeSynopsysAdapterError(
          SynopsysAdapterFailureKind::MissingSemanticInput,
          descriptor.implementationSemanticIdentity,
          "materialized semantic input lacks its source Artifact");
    if (llvm::Error error = validateBundleInputPath(
            descriptor.implementationSemanticIdentity, file.relativePath))
      return error;
  }
  return llvm::Error::success();
}

external_tool::ExternalToolInvocationImportExpectation
makeExpectation(const SynopsysInvocationDescriptor &descriptor,
                const SynopsysBundleInputs &inputs) {
  external_tool::ExternalToolInvocationImportExpectation expectation;
  expectation.semanticContract = inputs.semanticContract;
  for (const external_tool::MaterializedBundleFile &file :
       inputs.semanticInputs) {
    expectation.semanticInputs.push_back(
        external_tool::ExternalToolInvocationSemanticInput{
            file.relativePath, *file.sourceArtifact,
            computeBlobDigest(llvm::ArrayRef<std::uint8_t>(
                reinterpret_cast<const std::uint8_t *>(file.contents.data()),
                file.contents.size()))});
  }
  for (const external_tool::ResolvedExternalFile &file :
       inputs.frozen.externalFiles)
    expectation.externalInputs.push_back(
        external_tool::ExternalToolInvocationExternalInput{
            file.providerInputSlot, file.fingerprint});
  for (const external_tool::ResolvedExternalFileTree &tree :
       inputs.frozen.externalFileTrees)
    expectation.externalFileTrees.push_back(
        external_tool::ExternalToolInvocationExternalFileTree{
            tree.providerInputSlot, tree.members});
  for (llvm::StringLiteral output : descriptor.declaredOutputs)
    expectation.declaredOutputs.push_back(output.str());
  return expectation;
}

} // namespace

char SynopsysAdapterError::ID = 0;

void SynopsysAdapterError::log(llvm::raw_ostream &stream) const {
  stream << "synopsys_adapter[" << adapter_ << "]: " << detail_;
}

std::error_code SynopsysAdapterError::convertToErrorCode() const {
  return llvm::inconvertibleErrorCode();
}

llvm::Error makeSynopsysAdapterError(SynopsysAdapterFailureKind kind,
                                     llvm::StringRef adapter,
                                     const llvm::Twine &detail) {
  return llvm::make_error<SynopsysAdapterError>(kind, adapter.str(),
                                                detail.str());
}

bool acceptsSynopsysImplementationState(
    const SynopsysInvocationDescriptor &descriptor,
    hardware::RepresentationRootVariant variant,
    std::optional<hardware::RepresentationPhysicalStage> stage) {
  return llvm::is_contained(descriptor.acceptedStates,
                            SynopsysImplementationState{variant, stage});
}

llvm::Error validateSynopsysRepresentation(
    const SynopsysInvocationDescriptor &descriptor,
    const hardware::ImplementationRepresentationRoot &representation) {
  if (llvm::Error error =
          hardware::validateImplementationRepresentationRoot(representation))
    return makeSynopsysAdapterError(
        SynopsysAdapterFailureKind::UnsupportedImplementation,
        descriptor.implementationSemanticIdentity,
        llvm::toString(std::move(error)));
  if (!acceptsSynopsysImplementationState(descriptor, representation.variant,
                                          representation.stage))
    return makeSynopsysAdapterError(
        SynopsysAdapterFailureKind::UnsupportedImplementation,
        descriptor.implementationSemanticIdentity,
        "HardwareImplementation representation state is not accepted");

  const hardware::RepresentationFormatDescriptor &format =
      hardware::getRepresentationFormatDescriptor(representation.formatRef);
  if (llvm::Error error = hardware::validateRepresentationRootAdmission(
          format, representation)) {
    const SynopsysAdapterFailureKind kind =
        representation.variant ==
                hardware::RepresentationRootVariant::AsicPhysical
            ? SynopsysAdapterFailureKind::PublicationUnavailable
            : SynopsysAdapterFailureKind::UnsupportedImplementation;
    return makeSynopsysAdapterError(kind,
                                    descriptor.implementationSemanticIdentity,
                                    llvm::toString(std::move(error)));
  }

  if (descriptor.requiresGenerationConstraint &&
      llvm::none_of(representation.payloads, [](const auto &payload) {
        return payload.role == hardware::PayloadRole::GenerationConstraint;
      }))
    return makeSynopsysAdapterError(
        SynopsysAdapterFailureKind::MissingSemanticInput,
        descriptor.implementationSemanticIdentity,
        "HardwareImplementation has no GenerationConstraint payload");
  return llvm::Error::success();
}

llvm::Error validateSynopsysProviderInputs(
    const SynopsysInvocationDescriptor &descriptor,
    llvm::ArrayRef<external_tool::ResolvedExternalFile> files,
    llvm::ArrayRef<external_tool::ResolvedExternalFileTree> fileTrees) {
  std::vector<std::string> actual;
  actual.reserve(files.size() + fileTrees.size());
  for (const external_tool::ResolvedExternalFile &file : files)
    actual.push_back(file.providerInputSlot);
  for (const external_tool::ResolvedExternalFileTree &tree : fileTrees)
    actual.push_back(tree.providerInputSlot);
  llvm::sort(actual);
  if (std::adjacent_find(actual.begin(), actual.end()) != actual.end())
    return makeSynopsysAdapterError(
        SynopsysAdapterFailureKind::DescriptorMismatch,
        descriptor.implementationSemanticIdentity,
        "provider input slot is duplicated");

  std::vector<std::string> required;
  required.reserve(descriptor.requiredProviderInputs.size());
  for (llvm::StringLiteral slot : descriptor.requiredProviderInputs)
    required.push_back(slot.str());
  llvm::sort(required);
  if (actual != required)
    return makeSynopsysAdapterError(
        SynopsysAdapterFailureKind::MissingProviderInput,
        descriptor.implementationSemanticIdentity,
        "resolved provider input slots do not match the descriptor");
  return llvm::Error::success();
}

llvm::Error
validateSynopsysSemanticInputs(const SynopsysInvocationDescriptor &descriptor,
                               const SynopsysBundleInputs &inputs,
                               llvm::ArrayRef<std::string> requiredPaths) {
  std::set<std::string> materialized;
  for (const external_tool::MaterializedBundleFile &input :
       inputs.semanticInputs) {
    if (!materialized.insert(input.relativePath).second)
      return makeSynopsysAdapterError(
          SynopsysAdapterFailureKind::DescriptorMismatch,
          descriptor.implementationSemanticIdentity,
          "materialized semantic input path is duplicated");
  }

  std::set<std::string> required;
  for (const std::string &path : requiredPaths) {
    if (!required.insert(path).second)
      return makeSynopsysAdapterError(
          SynopsysAdapterFailureKind::DescriptorMismatch,
          descriptor.implementationSemanticIdentity,
          "required semantic input is duplicated");
    if (materialized.find(path) == materialized.end())
      return makeSynopsysAdapterError(
          SynopsysAdapterFailureKind::MissingSemanticInput,
          descriptor.implementationSemanticIdentity,
          "required semantic input '" + path + "' is not materialized");
  }
  return llvm::Error::success();
}

llvm::Expected<external_tool::ExternalToolInvocationBundleSpec>
makeSynopsysInvocationBundleSpec(
    const SynopsysInvocationDescriptor &descriptor,
    const SynopsysBundleInputs &inputs,
    std::vector<std::vector<std::string>> commands,
    std::vector<external_tool::MaterializedBundleFile> drivers) {
  if (llvm::Error error = validateBundleInputs(descriptor, inputs))
    return std::move(error);
  for (const external_tool::MaterializedBundleFile &driver : drivers)
    if (driver.sourceArtifact ||
        !llvm::StringRef(driver.relativePath).starts_with("drivers/"))
      return makeSynopsysAdapterError(
          SynopsysAdapterFailureKind::DescriptorMismatch,
          descriptor.implementationSemanticIdentity,
          "generated driver has invalid ownership or placement");

  std::vector<external_tool::MaterializedBundleFile> files = std::move(drivers);
  files.insert(files.end(), inputs.semanticInputs.begin(),
               inputs.semanticInputs.end());
  std::vector<std::string> outputs;
  outputs.reserve(descriptor.declaredOutputs.size());
  for (llvm::StringLiteral output : descriptor.declaredOutputs)
    outputs.push_back(output.str());

  return external_tool::ExternalToolInvocationBundleSpec{
      inputs.semanticContract,
      inputs.frozen.tool,
      inputs.frozen.toolVersionProbe,
      inputs.frozen.runtime,
      inputs.frozen.containerVersionProbe,
      std::move(commands),
      inputs.frozen.inheritEnvironment,
      std::move(outputs),
      std::move(files),
      inputs.frozen.externalFiles,
      inputs.frozen.externalFileTrees};
}

llvm::Expected<external_tool::ImportedExternalToolInvocationBundle>
importSynopsysInvocation(
    const SynopsysInvocationDescriptor &descriptor,
    const external_tool::PreparedExternalToolInvocation &prepared,
    const SynopsysBundleInputs &inputs) {
  if (llvm::Error error = validateBundleInputs(descriptor, inputs))
    return std::move(error);

  auto attempt = external_tool::importExternalToolInvocationAttempt(
      prepared, makeExpectation(descriptor, inputs));
  if (!attempt)
    return mapIntegrityError(descriptor, attempt.takeError());
  if (std::holds_alternative<
          external_tool::IncompleteExternalToolInvocationAttempt>(*attempt))
    return llvm::make_error<
        external_tool::IncompleteExternalToolInvocationError>();

  if (std::holds_alternative<
          external_tool::ImportedExternalToolInvocationBundle>(*attempt))
    return std::get<external_tool::ImportedExternalToolInvocationBundle>(
        std::move(*attempt));

  const auto &failed =
      std::get<external_tool::FailedExternalToolInvocationAttempt>(*attempt);

  using Status = external_tool::InvocationCompletionStatus;
  SynopsysAdapterFailureKind failure;
  switch (failed.status) {
  case Status::Success:
    return makeSynopsysAdapterError(
        SynopsysAdapterFailureKind::IntegrityFailure,
        descriptor.implementationSemanticIdentity,
        "failed invocation outcome carries success status");
  case Status::MissingEnvironment:
  case Status::ModuleActivationFailed:
    failure = SynopsysAdapterFailureKind::ActivationUnavailable;
    break;
  case Status::VersionMismatch:
    failure = SynopsysAdapterFailureKind::IncompatibleVersion;
    break;
  case Status::BundleContentMismatch:
    failure = SynopsysAdapterFailureKind::IntegrityFailure;
    break;
  case Status::ToolExit:
    failure = SynopsysAdapterFailureKind::ToolExecutionFailed;
    break;
  case Status::MissingOutput:
    failure = SynopsysAdapterFailureKind::MissingDeclaredOutput;
    break;
  }
  return makeSynopsysAdapterError(
      failure, descriptor.implementationSemanticIdentity,
      "invocation completion status is not successful (exit code " +
          llvm::Twine(failed.exitCode) + ")");
}

llvm::Error
validateSynopsysOutputInventory(const SynopsysInvocationDescriptor &descriptor,
                                llvm::StringRef bundleRoot) {
  std::set<std::string> allowed{"completion.json", "stderr.log", "stdout.log"};
  for (llvm::StringLiteral output : descriptor.declaredOutputs) {
    const std::filesystem::path path(output.str());
    if (path.parent_path() != "outputs" || path.filename().empty())
      return makeSynopsysAdapterError(
          SynopsysAdapterFailureKind::DescriptorMismatch,
          descriptor.implementationSemanticIdentity,
          "declared output is not a direct member of outputs");
    allowed.insert(path.filename().string());
  }

  const std::filesystem::path outputs =
      std::filesystem::path(bundleRoot.str()) / "outputs";
  std::set<std::string> found;
  std::error_code error;
  const std::filesystem::file_status rootStatus =
      std::filesystem::symlink_status(outputs, error);
  if (error || !std::filesystem::is_directory(rootStatus) ||
      std::filesystem::is_symlink(rootStatus))
    return makeSynopsysAdapterError(
        SynopsysAdapterFailureKind::IntegrityFailure,
        descriptor.implementationSemanticIdentity,
        "outputs directory is missing or not an ordinary directory");
  for (std::filesystem::directory_iterator iterator(outputs, error), end;
       !error && iterator != end; iterator.increment(error)) {
    const std::filesystem::path path = iterator->path();
    const std::filesystem::file_status status =
        std::filesystem::symlink_status(path, error);
    if (error)
      break;
    const std::string name = path.filename().string();
    if (!std::filesystem::is_regular_file(status) ||
        std::filesystem::is_symlink(status) || !allowed.count(name))
      return makeSynopsysAdapterError(
          SynopsysAdapterFailureKind::IntegrityFailure,
          descriptor.implementationSemanticIdentity,
          "outputs directory contains undeclared output '" + name + "'");
    found.insert(name);
  }
  if (error)
    return makeSynopsysAdapterError(
        SynopsysAdapterFailureKind::IntegrityFailure,
        descriptor.implementationSemanticIdentity,
        "could not enumerate outputs directory: " + error.message());
  if (found != allowed)
    return makeSynopsysAdapterError(
        SynopsysAdapterFailureKind::IntegrityFailure,
        descriptor.implementationSemanticIdentity,
        "outputs directory omits a lifecycle or declared output");
  return llvm::Error::success();
}

llvm::Expected<std::string> readSynopsysDeclaredOutput(
    const SynopsysInvocationDescriptor &descriptor,
    const external_tool::ImportedExternalToolInvocationBundle &bundle,
    llvm::StringRef relativePath) {
  const bool declared =
      llvm::any_of(descriptor.declaredOutputs, [&](llvm::StringLiteral output) {
        return output == relativePath;
      });
  if (!declared)
    return makeSynopsysAdapterError(
        SynopsysAdapterFailureKind::DescriptorMismatch,
        descriptor.implementationSemanticIdentity,
        "importer requested an undeclared output '" + relativePath + "'");
  auto contents = external_tool::readExternalToolInvocationDeclaredOutput(
      bundle, relativePath);
  if (!contents)
    return mapIntegrityError(descriptor, contents.takeError());
  return std::move(*contents);
}

bool isPortableHdlIdentifier(llvm::StringRef value) {
  const auto first = [](char character) {
    return (character >= 'A' && character <= 'Z') ||
           (character >= 'a' && character <= 'z') || character == '_';
  };
  const auto rest = [&](char character) {
    return first(character) || (character >= '0' && character <= '9') ||
           character == '$';
  };
  return !value.empty() && first(value.front()) &&
         llvm::all_of(value.drop_front(), rest);
}

llvm::Error validateBundleInputPath(llvm::StringRef adapter,
                                    llvm::StringRef path) {
  if (path.empty() || path.contains('\0'))
    return makeSynopsysAdapterError(
        SynopsysAdapterFailureKind::MissingSemanticInput, adapter,
        "bundle input path is empty or contains NUL");
  const std::filesystem::path candidate(path.str());
  if (candidate.is_absolute() || candidate.lexically_normal() != candidate ||
      !llvm::StringRef(candidate.generic_string()).starts_with("inputs/"))
    return makeSynopsysAdapterError(
        SynopsysAdapterFailureKind::MissingSemanticInput, adapter,
        "bundle input path must be normalized beneath inputs");
  return llvm::Error::success();
}

llvm::Expected<std::string> renderTclWord(llvm::StringRef adapter,
                                          llvm::StringRef value) {
  if (value.empty() || value.contains('\0') || value.contains('{') ||
      value.contains('}') || value.contains('\n') || value.contains('\r') ||
      value.contains("\\\n"))
    return makeSynopsysAdapterError(
        SynopsysAdapterFailureKind::MissingSemanticInput, adapter,
        "value cannot be represented as one literal Tcl word");
  return "{" + value.str() + "}";
}

std::string renderSynopsysTclBatch(llvm::StringRef commands,
                                   llvm::StringRef publication) {
  const llvm::StringLiteral requireClean =
      "if {[get_message_info -error_count] != 0} {\n"
      "  error {Synopsys tool emitted error diagnostics}\n"
      "}\n";
  std::string driver = "proc loom_main {} {\n";
  driver += commands;
  driver += requireClean;
  driver += publication;
  driver += requireClean;
  driver += "}\n"
            "if {[catch {loom_main} loom_error]} {\n"
            "  puts stderr $loom_error\n"
            "  exit 1\n"
            "}\n"
            "exit 0\n";
  return driver;
}

llvm::Expected<evaluation::DecimalValue>
parseSynopsysDecimal(llvm::StringRef adapter, llvm::StringRef field,
                     llvm::StringRef value, bool permitZero) {
  auto invalid = [&](const llvm::Twine &detail)
      -> llvm::Expected<evaluation::DecimalValue> {
    return makeSynopsysAdapterError(
        SynopsysAdapterFailureKind::ParserFailure, adapter,
        field + " is not a finite decimal: " + detail);
  };
  if (value.empty() || value.front() == '+' || value.front() == '-')
    return invalid("sign or digits are invalid");

  const std::size_t exponentPosition = value.find_first_of("eE");
  if (exponentPosition != llvm::StringRef::npos &&
      value.drop_front(exponentPosition + 1).find_first_of("eE") !=
          llvm::StringRef::npos)
    return invalid("multiple exponents");
  const llvm::StringRef mantissa = value.take_front(exponentPosition);
  const llvm::StringRef exponentText =
      exponentPosition == llvm::StringRef::npos
          ? llvm::StringRef()
          : value.drop_front(exponentPosition + 1);

  std::int64_t explicitExponent = 0;
  if (!exponentText.empty()) {
    const char *begin = exponentText.begin();
    const char *end = exponentText.end();
    const auto parsed = std::from_chars(begin, end, explicitExponent);
    if (parsed.ec != std::errc() || parsed.ptr != end)
      return invalid("exponent is invalid");
  } else if (exponentPosition != llvm::StringRef::npos) {
    return invalid("exponent is empty");
  }

  const std::size_t dot = mantissa.find('.');
  if (dot != llvm::StringRef::npos &&
      mantissa.drop_front(dot + 1).contains('.'))
    return invalid("multiple decimal points");
  const llvm::StringRef integer = mantissa.take_front(dot);
  const llvm::StringRef fraction = dot == llvm::StringRef::npos
                                       ? llvm::StringRef()
                                       : mantissa.drop_front(dot + 1);
  if (integer.empty() || (dot != llvm::StringRef::npos && fraction.empty()))
    return invalid("mantissa is incomplete");
  const auto digitsOnly = [](llvm::StringRef digits) {
    return llvm::all_of(digits, [](char c) { return c >= '0' && c <= '9'; });
  };
  if (!digitsOnly(integer) || !digitsOnly(fraction))
    return invalid("mantissa contains a nondigit");

  std::string digits = (integer + fraction).str();
  const std::size_t nonzero = digits.find_first_not_of('0');
  if (nonzero == std::string::npos) {
    if (!permitZero)
      return invalid("zero is outside the value domain");
    return evaluation::DecimalValue::get(0, 0);
  }
  digits.erase(0, nonzero);
  std::int64_t coefficient = 0;
  const auto coefficientResult = std::from_chars(
      digits.data(), digits.data() + digits.size(), coefficient);
  if (coefficientResult.ec != std::errc() ||
      coefficientResult.ptr != digits.data() + digits.size())
    return invalid("coefficient is out of range");
  if (fraction.size() >
      static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()))
    return invalid("fraction is too long");
  const std::int64_t fractionDigits =
      static_cast<std::int64_t>(fraction.size());
  if (explicitExponent <
      std::numeric_limits<std::int64_t>::min() + fractionDigits)
    return invalid("exponent is out of range");
  return evaluation::DecimalValue::get(coefficient,
                                       explicitExponent - fractionDigits);
}

} // namespace loom::eda::synopsys
