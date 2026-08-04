#include "ImplementationPlatform/ImplementationPlatform.h"

#include "Common/ArtifactStore.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <utility>
#include <variant>
#include <vector>

using namespace loom;
using namespace loom::platform;

namespace {

[[noreturn]] void fail(llvm::StringRef test, const std::string &message) {
  std::cerr << test.str() << ": " << message << '\n';
  std::exit(1);
}

void require(llvm::StringRef test, bool condition,
             const std::string &message) {
  if (!condition)
    fail(test, message);
}

template <typename T>
T take(llvm::StringRef test, llvm::Expected<T> value) {
  if (!value)
    fail(test, llvm::toString(value.takeError()));
  return std::move(*value);
}

template <typename T>
void expectErrorContains(llvm::StringRef test, llvm::Expected<T> value,
                         llvm::StringRef expected) {
  if (value)
    fail(test, "expected an error");
  const std::string message = llvm::toString(value.takeError());
  require(test, llvm::StringRef(message).contains(expected), message);
}

ImplementationPlatformDraft asicDraft() {
  return ImplementationPlatformDraft{
      AsicTarget{"saed14", "EDK_08_2025"},
      {"ss_0p72v_125c", "tt_0p80v_25c"}};
}

void asicCornersAreCanonical(const ArtifactStore &store) {
  const llvm::StringRef test = __func__;
  FinalizedImplementationPlatform first =
      take(test, finalizeImplementationPlatform(asicDraft(), store));

  ImplementationPlatformDraft reordered = asicDraft();
  std::reverse(reordered.technologyCornerKeys.begin(),
               reordered.technologyCornerKeys.end());
  FinalizedImplementationPlatform second =
      take(test, finalizeImplementationPlatform(std::move(reordered), store));

  require(test, first.reference() == second.reference(),
          "corner authoring order changed platform identity");
  require(test,
          first.canonicalBytes().bytes().equals(second.canonicalBytes().bytes()),
          "corner authoring order changed canonical bytes");

  const auto *target = std::get_if<AsicTarget>(&first.platform().target());
  require(test, target && target->technologyIdentity == "saed14" &&
                    target->releaseIdentity == "EDK_08_2025",
          "ASIC target was not preserved");

  const auto corners = first.platform().technologyCorners();
  require(test, corners.size() == 2, "corner catalog has the wrong size");
  require(test, corners[0].id == TechnologyCornerId(0) &&
                    corners[0].key == "ss_0p72v_125c" &&
                    corners[1].id == TechnologyCornerId(1) &&
                    corners[1].key == "tt_0p80v_25c",
          "corner catalog is not dense canonical key order");

  FinalizedImplementationPlatform imported =
      take(test, importImplementationPlatform(first.reference(), store));
  require(test, imported.reference() == first.reference() &&
                    imported.canonicalBytes().bytes().equals(
                        first.canonicalBytes().bytes()),
          "platform roundtrip changed canonical content");

  TechnologyCorner resolved = take(
      test, resolveTechnologyCorner(
                TechnologyCornerRef{first.reference().artifact,
                                    TechnologyCornerId(1)},
                store));
  require(test, resolved.key == "tt_0p80v_25c",
          "typed corner reference resolved to the wrong key");
  expectErrorContains(
      test,
      resolveTechnologyCorner(
          TechnologyCornerRef{first.reference().artifact,
                              TechnologyCornerId(2)},
          store),
      "out of range");
}

void fpgaTargetIsClosed(const ArtifactStore &store) {
  const llvm::StringRef test = __func__;
  ImplementationPlatformDraft draft{
      FpgaTarget{FpgaVendor::AmdXilinx, "xcvh1782-lsva4737-3HP-e-S"},
      {"speed_grade_3"}};
  FinalizedImplementationPlatform finalized =
      take(test, finalizeImplementationPlatform(std::move(draft), store));
  const auto *target = std::get_if<FpgaTarget>(&finalized.platform().target());
  require(test, target && target->vendor == FpgaVendor::AmdXilinx &&
                    target->deviceOrderingCode ==
                        "xcvh1782-lsva4737-3HP-e-S",
          "FPGA target was not preserved");

  const llvm::ArrayRef<std::uint8_t> bytes = finalized.canonicalBytes().bytes();
  const llvm::StringRef json(reinterpret_cast<const char *>(bytes.data()),
                             bytes.size());
  require(test, json.contains("\"device_ordering_code\"") &&
                    !json.contains("\"package\"") &&
                    !json.contains("\"speed_grade\"") &&
                    !json.contains("payload"),
          "FPGA root copied derived or payload fields");
}

void semanticTargetChangesIdentity(const ArtifactStore &store) {
  const llvm::StringRef test = __func__;
  ImplementationPlatformDraft firstDraft = asicDraft();
  ImplementationPlatformDraft secondDraft = asicDraft();
  std::get<AsicTarget>(secondDraft.target).releaseIdentity = "EDK_09_2025";
  FinalizedImplementationPlatform first =
      take(test, finalizeImplementationPlatform(std::move(firstDraft), store));
  FinalizedImplementationPlatform second =
      take(test, finalizeImplementationPlatform(std::move(secondDraft), store));
  require(test, first.reference().artifact != second.reference().artifact,
          "ASIC release change did not change platform identity");
}

void invalidDraftsAreRejected(const ArtifactStore &store) {
  const llvm::StringRef test = __func__;
  ImplementationPlatformDraft emptyCorners = asicDraft();
  emptyCorners.technologyCornerKeys.clear();
  expectErrorContains(
      test, finalizeImplementationPlatform(std::move(emptyCorners), store),
      "nonempty");

  ImplementationPlatformDraft duplicateCorners = asicDraft();
  duplicateCorners.technologyCornerKeys = {"typical", "typical"};
  expectErrorContains(
      test, finalizeImplementationPlatform(std::move(duplicateCorners), store),
      "duplicate");

  ImplementationPlatformDraft malformedTechnology = asicDraft();
  std::get<AsicTarget>(malformedTechnology.target).technologyIdentity =
      "bad target";
  expectErrorContains(
      test,
      finalizeImplementationPlatform(std::move(malformedTechnology), store),
      "technology_identity");

  ImplementationPlatformDraft malformedCorner = asicDraft();
  malformedCorner.technologyCornerKeys = {"_bad"};
  expectErrorContains(
      test, finalizeImplementationPlatform(std::move(malformedCorner), store),
      "corner_key");

  ImplementationPlatformDraft unknownVendor{
      FpgaTarget{static_cast<FpgaVendor>(99), "part-1"}, {"default"}};
  expectErrorContains(
      test, finalizeImplementationPlatform(std::move(unknownVendor), store),
      "FPGA vendor");
}

void noncanonicalStoredRootIsRejected(const ArtifactStore &store) {
  const llvm::StringRef test = __func__;
  FinalizedImplementationPlatform finalized =
      take(test, finalizeImplementationPlatform(asicDraft(), store));
  std::vector<std::uint8_t> bytes(finalized.canonicalBytes().bytes().begin(),
                                  finalized.canonicalBytes().bytes().end());
  bytes.push_back('\n');
  ArtifactIdentity identity = take(
      test, store.put(implementationPlatformSchema,
                      CanonicalSemanticBytes(std::move(bytes))));
  ArtifactRootReference reference{implementationPlatformSchema.identity.str(),
                                  implementationPlatformSchema.version,
                                  std::move(identity)};
  expectErrorContains(test, importImplementationPlatform(reference, store),
                      "not canonical");
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2)
    fail("main", "expected one artifact-store root");
  const std::filesystem::path root =
      std::filesystem::absolute(argv[1]).lexically_normal();
  std::filesystem::create_directories(root);
  const ArtifactStore store(root.string());
  asicCornersAreCanonical(store);
  fpgaTargetIsClosed(store);
  semanticTargetChangesIdentity(store);
  invalidDraftsAreRejected(store);
  noncanonicalStoredRootIsRejected(store);
  return 0;
}
