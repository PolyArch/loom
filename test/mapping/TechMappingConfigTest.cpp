#include "Mapping/Tech/TechMappingConfig.h"
#include "Common/ComponentViewDigest.h"
#include "Config/ResolvedConfig.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <utility>

namespace {

[[noreturn]] void fail(llvm::StringRef message) {
  llvm::errs() << "tech mapping config test: " << message << '\n';
  std::exit(1);
}

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

bool rejected(
    llvm::Expected<loom::mapping::ResolvedTechMappingConfigView> value) {
  if (value)
    return false;
  llvm::consumeError(value.takeError());
  return true;
}

std::uint64_t readU64(llvm::ArrayRef<std::uint8_t> bytes) {
  if (bytes.size() != 8)
    fail("u64 field has the wrong width");
  std::uint64_t value = 0;
  for (std::uint8_t byte : bytes)
    value = (value << 8) | byte;
  return value;
}

void checkProjectionAndAdoption() {
  loom::ResolvedConfig config = loom::defaultResolvedConfig();
  config.dse.techMapping.matchRowAttemptLimit = 17;
  config.dse.techMapping.partialCoverExpansionLimit = 23;
  config.dse.techMapping.candidateEvaluationLimit = 29;
  config.dse.techMapping.candidatePublicationLimit = 5;

  const auto view =
      take(loom::mapping::projectResolvedTechMappingConfigView(config));
  if (view.matchRowAttemptLimit() != 17 ||
      view.partialCoverExpansionLimit() != 23 ||
      view.candidateEvaluationLimit() != 29 ||
      view.candidatePublicationLimit() != 5)
    fail("projector changed a TechMapping limit");

  const llvm::StringRef descriptor = "loom.tech_mapping.config.2.0";
  if (view.schemaDescriptorBytes() !=
      llvm::ArrayRef<std::uint8_t>(
          reinterpret_cast<const std::uint8_t *>(descriptor.data()),
          descriptor.size()))
    fail("schema descriptor is not the exact 2.0 spelling");
  const auto bytes = view.canonicalViewBytes();
  if (bytes.size() != 32 || readU64(bytes.slice(0, 8)) != 17 ||
      readU64(bytes.slice(8, 8)) != 23 || readU64(bytes.slice(16, 8)) != 29 ||
      readU64(bytes.slice(24, 8)) != 5)
    fail("canonical view is not the four ordered u64be fields");

  const auto expectedDigest = take(loom::computeComponentViewDigest(
      view.schemaDescriptorBytes(), view.canonicalViewBytes()));
  if (view.digest() != expectedDigest)
    fail("view did not use the Common component digest");
  const auto adopted = take(loom::mapping::adoptResolvedTechMappingConfigView(
      view.schemaDescriptorBytes(), view.canonicalViewBytes(), view.digest()));
  if (adopted.canonicalViewBytes() != view.canonicalViewBytes())
    fail("adoption changed canonical bytes");
}

void checkInvalidWire() {
  const auto view = take(loom::mapping::projectResolvedTechMappingConfigView(
      loom::defaultResolvedConfig()));
  std::array<std::uint8_t, 32> bytes{};
  const auto digest = take(
      loom::computeComponentViewDigest(view.schemaDescriptorBytes(), bytes));
  if (!rejected(loom::mapping::adoptResolvedTechMappingConfigView(
          view.schemaDescriptorBytes(), bytes, digest)))
    fail("zero semantic limit was adopted");
}

} // namespace

int main() {
  checkProjectionAndAdoption();
  checkInvalidWire();
  llvm::outs() << "tech mapping config tests passed\n";
  return 0;
}
