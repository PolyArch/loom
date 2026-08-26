#include "InvocationBundleInternal.h"

#include "Common/BlobDigest.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace loom::external_tool {

llvm::Error invocationBundleError(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "invocation_bundle_invalid: " + message);
}

llvm::StringRef completionStatusSpelling(InvocationCompletionStatus status) {
  switch (status) {
  case InvocationCompletionStatus::Success:
    return "success";
  case InvocationCompletionStatus::MissingEnvironment:
    return "missing_environment";
  case InvocationCompletionStatus::ModuleActivationFailed:
    return "module_activation_failed";
  case InvocationCompletionStatus::VersionMismatch:
    return "version_mismatch";
  case InvocationCompletionStatus::BundleContentMismatch:
    return "bundle_content_mismatch";
  case InvocationCompletionStatus::ToolExit:
    return "tool_exit";
  case InvocationCompletionStatus::MissingOutput:
    return "missing_output";
  }
  llvm_unreachable("closed invocation completion status");
}

namespace {

std::optional<InvocationCompletionStatus>
parseCompletionStatus(llvm::StringRef spelling) {
  constexpr std::array statuses{
      InvocationCompletionStatus::Success,
      InvocationCompletionStatus::MissingEnvironment,
      InvocationCompletionStatus::ModuleActivationFailed,
      InvocationCompletionStatus::VersionMismatch,
      InvocationCompletionStatus::BundleContentMismatch,
      InvocationCompletionStatus::ToolExit,
      InvocationCompletionStatus::MissingOutput};
  for (InvocationCompletionStatus status : statuses)
    if (spelling == completionStatusSpelling(status))
      return status;
  return std::nullopt;
}

} // namespace

std::string
serializeInvocationCompletion(InvocationCompletionStatus status, int exitCode,
                              const BlobDigest &manifestDigest,
                              const BlobDigest &attemptToken,
                              llvm::ArrayRef<BlobDigest> outputDigests) {
  std::string canonical =
      "{\"schema\":\"" + kInvocationCompletionSchema.str() +
      "\",\"version\":\"" + kInvocationCompletionVersion.str() +
      "\",\"status\":\"" + completionStatusSpelling(status).str() +
      "\",\"exit_code\":" + std::to_string(exitCode) +
      ",\"manifest_sha256\":\"" + formatBlobDigestHex(manifestDigest) +
      "\",\"attempt_sha256\":\"" + formatBlobDigestHex(attemptToken) +
      "\",\"output_sha256\":[";
  for (std::size_t index = 0; index < outputDigests.size(); ++index) {
    if (index != 0)
      canonical += ',';
    canonical += "\"" + formatBlobDigestHex(outputDigests[index]) + "\"";
  }
  canonical += "]}\n";
  return canonical;
}

llvm::Expected<InvocationCompletion>
parseInvocationCompletion(llvm::StringRef contents) {
  llvm::Expected<llvm::json::Value> parsed = llvm::json::parse(contents);
  if (!parsed)
    return invocationBundleError("completion record is malformed: " +
                                 llvm::toString(parsed.takeError()));
  const llvm::json::Object *object = parsed->getAsObject();
  if (!object || object->size() != 7)
    return invocationBundleError("completion record has an invalid shape");
  const std::optional<llvm::StringRef> schema = object->getString("schema");
  const std::optional<llvm::StringRef> version = object->getString("version");
  const std::optional<llvm::StringRef> status = object->getString("status");
  const std::optional<std::int64_t> exitCode = object->getInteger("exit_code");
  const std::optional<llvm::StringRef> manifestText =
      object->getString("manifest_sha256");
  const std::optional<llvm::StringRef> attemptText =
      object->getString("attempt_sha256");
  const llvm::json::Array *outputArray = object->getArray("output_sha256");
  if (!schema || *schema != kInvocationCompletionSchema || !version ||
      *version != kInvocationCompletionVersion || !status || !exitCode ||
      *exitCode < 0 || *exitCode > 255 || !manifestText || !attemptText ||
      !outputArray)
    return invocationBundleError("completion record fields are invalid");
  std::optional<InvocationCompletionStatus> parsedStatus =
      parseCompletionStatus(*status);
  if (!parsedStatus ||
      ((*parsedStatus == InvocationCompletionStatus::Success) !=
       (*exitCode == 0)))
    return invocationBundleError(
        "completion status and exit code are inconsistent");
  auto manifestDigest = parseBlobDigestHex(*manifestText);
  if (!manifestDigest)
    return manifestDigest.takeError();
  auto attemptToken = parseBlobDigestHex(*attemptText);
  if (!attemptToken)
    return attemptToken.takeError();
  std::vector<BlobDigest> outputDigests;
  outputDigests.reserve(outputArray->size());
  for (const llvm::json::Value &value : *outputArray) {
    std::optional<llvm::StringRef> digestText = value.getAsString();
    if (!digestText)
      return invocationBundleError("completion output digest must be a string");
    auto digest = parseBlobDigestHex(*digestText);
    if (!digest)
      return digest.takeError();
    outputDigests.push_back(std::move(*digest));
  }
  if (*parsedStatus != InvocationCompletionStatus::Success &&
      !outputDigests.empty())
    return invocationBundleError(
        "failed completion record contains output digests");
  if (contents != serializeInvocationCompletion(
                      *parsedStatus, static_cast<int>(*exitCode),
                      *manifestDigest, *attemptToken, outputDigests))
    return invocationBundleError("completion record is not canonical");
  return InvocationCompletion{
      *parsedStatus, static_cast<int>(*exitCode), std::move(*manifestDigest),
      std::move(*attemptToken), std::move(outputDigests)};
}

llvm::Error validateInvocationCompletionExecutionBoundary(
    const PreparedExternalToolInvocation &prepared,
    const BlobDigest &attemptToken, int exitCode,
    const std::optional<InvocationCompletion> &completion) {
  if (!completion)
    return llvm::Error::success();
  if (completion->manifestDigest != prepared.manifestDigest)
    return invocationBundleError(
        "completion does not bind the observed invocation manifest");
  if (completion->attemptToken != attemptToken)
    return invocationBundleError(
        "completion does not bind the observed attempt generation");
  if (completion->exitCode != exitCode)
    return invocationBundleError(
        "completion does not match the observed execution exit code");
  return llvm::Error::success();
}

} // namespace loom::external_tool
