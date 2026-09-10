#include "Runtime/Gem5BuildReadiness.h"

#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"

#include <filesystem>
#include <system_error>

namespace loom::runtime {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "gem5_build_readiness_invalid: " + message);
}

} // namespace

llvm::Expected<Gem5BuildReadiness>
readGem5BuildReadiness(llvm::StringRef path) {
  std::error_code error;
  const auto canonicalPath = std::filesystem::canonical(path.str(), error);
  if (error)
    return invalid("cannot resolve readiness stamp: " + error.message());
  auto contents = llvm::MemoryBuffer::getFile(canonicalPath.string());
  if (!contents)
    return invalid("cannot read readiness stamp: " +
                   contents.getError().message());
  auto value = llvm::json::parse((*contents)->getBuffer());
  if (!value)
    return invalid("readiness stamp is not valid JSON: " +
                   llvm::toString(value.takeError()));
  const auto *object = value->getAsObject();
  if (!object)
    return invalid("readiness stamp is not an object");
  const auto schema = object->getString("schema");
  const auto bridgeAbi = object->getString("bridge_abi_identity");
  const auto repository = object->getString("gem5_repository_identity");
  const auto commit = object->getString("gem5_full_commit_identity");
  const auto configuration = object->getString("build_configuration_digest");
  const auto binary = object->getString("binary");
  const auto binarySha = object->getString("binary_sha256");
  const auto versionProbe = object->getString("version_probe");
  if (!schema || *schema != "loom.gem5_build_readiness.1" || !bridgeAbi ||
      !repository || !commit || !configuration || !binary || !binarySha ||
      !versionProbe)
    return invalid("readiness stamp has a foreign schema or omits an identity "
                   "field");
  return Gem5BuildReadiness{{repository->str(), commit->str(),
                             configuration->str(), binarySha->str()},
                            canonicalPath.string(),
                            binary->str(),
                            bridgeAbi->str(),
                            versionProbe->str()};
}

} // namespace loom::runtime
