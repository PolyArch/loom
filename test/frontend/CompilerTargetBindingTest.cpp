#include "Frontend/Executable/CompilerTargetBinding.h"
#include "ADG/Builtin.h"
#include "Common/ArtifactStore.h"
#include "Fabric/Artifact/FabricSystemRootView.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace {

[[noreturn]] void fail(llvm::StringRef test, const std::string &message) {
  llvm::errs() << test << ": " << message << '\n';
  std::exit(1);
}

void require(llvm::StringRef test, bool condition, llvm::StringRef message) {
  if (!condition)
    fail(test, message.str());
}

template <typename T> T take(llvm::StringRef test, llvm::Expected<T> value) {
  if (!value)
    fail(test, llvm::toString(value.takeError()));
  return std::move(*value);
}

void expectError(llvm::StringRef test, llvm::Error error,
                 llvm::StringRef marker) {
  if (!error)
    fail(test, "accepted a value that must fail closed");
  const std::string message = llvm::toString(std::move(error));
  if (!llvm::StringRef(message).contains(marker))
    fail(test, "expected '" + marker.str() + "' in: " + message);
}

template <typename T>
void expectError(llvm::StringRef test, llvm::Expected<T> value,
                 llvm::StringRef marker) {
  if (value)
    fail(test, "accepted a value that must fail closed");
  expectError(test, value.takeError(), marker);
}

class TemporaryDirectory final {
public:
  explicit TemporaryDirectory(llvm::StringRef test) {
    llvm::SmallString<128> path;
    std::error_code error =
        llvm::sys::fs::createUniqueDirectory("loom-target-binding", path);
    if (error)
      fail(test, error.message());
    path_ = path.str().str();
  }

  ~TemporaryDirectory() {
    std::error_code error = llvm::sys::fs::remove_directories(path_);
    if (error)
      llvm::errs() << "could not remove " << path_ << ": " << error.message()
                   << '\n';
  }

  llvm::StringRef path() const { return path_; }

private:
  std::string path_;
};

std::vector<loom::fabric::AccCoreOccurrenceRef>
accCores(const loom::fabric::FabricArtifactView &view) {
  std::vector<loom::fabric::AccCoreOccurrenceRef> result;
  for (std::uint64_t id = 0;; ++id) {
    const auto kind = view.entityKind(id);
    if (!kind)
      break;
    if (*kind == loom::fabric::FabricEntityKind::AccCoreOccurrence)
      result.emplace_back(id);
  }
  return result;
}

loom::fabric::HostCoreOccurrenceRef
hostCore(llvm::StringRef test, const loom::fabric::FabricArtifactView &view) {
  for (std::uint64_t id = 0;; ++id) {
    const auto kind = view.entityKind(id);
    if (!kind)
      break;
    if (*kind == loom::fabric::FabricEntityKind::HostCoreOccurrence)
      return loom::fabric::HostCoreOccurrenceRef(id);
  }
  fail(test, "builtin System has no HostCore");
}

loom::CompilerTargetPolicy policy() {
  return {loom::fabric::RiscVAbi::Lp64d,
          loom::fabric::RiscVCodeModel::MediumAny,
          loom::fabric::RelocationModel::Static,
          "generic-rv64",
          {}};
}

loom::fabric::InstructionCoreArchitecturalContract
architectureFor(llvm::StringRef test, loom::fabric::RiscVXLen xlen) {
  loom::fabric::RiscVArchitectureDeclaration declaration;
  declaration.xlen = xlen;
  declaration.base = loom::fabric::RiscVBase::I;
  declaration.extensions = {loom::fabric::RiscVExtension::M,
                            loom::fabric::RiscVExtension::A};
  declaration.endianness = loom::fabric::InstructionEndianness::Little;
  declaration.physicalAddressWidthBits =
      xlen == loom::fabric::RiscVXLen::X64 ? 48 : 32;
  declaration.privilegeModes = {loom::fabric::PrivilegeMode::Machine};
  declaration.abiCapabilities = {xlen == loom::fabric::RiscVXLen::X64
                                     ? loom::fabric::RiscVAbi::Lp64
                                     : loom::fabric::RiscVAbi::Ilp32};
  declaration.memoryOrdering = loom::fabric::RiscVMemoryOrdering::Rvwmo;
  declaration.syncScopes = {loom::fabric::InstructionSyncScope::Hart};
  declaration.codeModels = {loom::fabric::RiscVCodeModel::MediumAny};
  declaration.relocationModels = {loom::fabric::RelocationModel::Static};
  declaration.runtimeServices = {
      loom::fabric::InstructionRuntimeService::ThreadDispatch};
  return take(test, loom::fabric::InstructionCoreArchitecturalContract::create(
                        std::move(declaration)));
}

void architectureFingerprintTracksIsa() {
  const llvm::StringRef test = __func__;
  const auto rv64 = architectureFor(test, loom::fabric::RiscVXLen::X64);
  const auto rv32 = architectureFor(test, loom::fabric::RiscVXLen::X32);
  require(test,
          loom::computeArchitectureFingerprint(rv64) !=
              loom::computeArchitectureFingerprint(rv32),
          "an XLEN change preserved the architecture fingerprint");
}

void bindingRoundTripAndCompatibility() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  loom::ArtifactStore store(directory.path());
  auto design = take(test, loom::adg::buildBuiltinTarget(
                               store, loom::adg::BuiltinTargetPreset::Small));
  require(test, design.roots().size() == 1,
          "builtin target did not produce one System root");
  const auto &system = design.roots().front();
  const auto cores = accCores(system.view());
  require(test, cores.size() >= 2, "builtin System has too few AccCores");

  const auto first = loom::CompilerProcessorArchitectureRef::instruction(
      {system.reference().artifact,
       loom::fabric::InstructionCoreContextRef{cores[0]}});
  const auto second = loom::CompilerProcessorArchitectureRef::instruction(
      {system.reference().artifact,
       loom::fabric::InstructionCoreContextRef{cores[1]}});
  auto finalized =
      take(test, loom::resolveCompilerTargetBinding(first, policy(), store));
  if (finalized.binding().targetTriple() != "riscv64-unknown-unknown-elf" ||
      finalized.binding().dataLayout().empty() ||
      finalized.binding().backendAbi() != loom::fabric::RiscVAbi::Lp64d)
    fail(test, "resolved binding lost its exact RISC-V target: triple='" +
                   finalized.binding().targetTriple().str() + "', layout='" +
                   finalized.binding().dataLayout().str() + "'");
  require(test,
          finalized.binding().compilerProvider().fullCommitIdentity.size() ==
              40,
          "resolved binding did not retain the pinned LLVM provider");
  if (llvm::Error error = loom::requireCompilerTargetCompatibility(
          finalized.binding(), second, store))
    fail(test, llvm::toString(std::move(error)));

  auto imported = take(
      test, loom::importCompilerTargetBinding(finalized.reference(), store));
  require(test,
          imported.canonicalBytes().bytes() ==
                  finalized.canonicalBytes().bytes() &&
              imported.binding().architectureFingerprint() ==
                  finalized.binding().architectureFingerprint(),
          "strict import changed binding semantics");

  const auto host = loom::CompilerProcessorArchitectureRef::host(
      {system.reference().artifact, hostCore(test, system.view())});
  expectError(test,
              loom::requireCompilerTargetCompatibility(finalized.binding(),
                                                       host, store),
              "processor_kind_mismatch");
}

void invalidAbiAndTamperedFingerprintFailClosed() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  loom::ArtifactStore store(directory.path());
  auto design = take(test, loom::adg::buildBuiltinTarget(
                               store, loom::adg::BuiltinTargetPreset::Small));
  const auto &system = design.roots().front();
  const auto cores = accCores(system.view());
  const auto processor = loom::CompilerProcessorArchitectureRef::instruction(
      {system.reference().artifact,
       loom::fabric::InstructionCoreContextRef{cores.front()}});

  loom::CompilerTargetPolicy wrongAbi = policy();
  wrongAbi.backendAbi = loom::fabric::RiscVAbi::Ilp32;
  expectError(test,
              loom::resolveCompilerTargetBinding(processor, wrongAbi, store),
              "backend_abi_not_admitted");

  auto finalized = take(
      test, loom::resolveCompilerTargetBinding(processor, policy(), store));
  std::string tampered(
      reinterpret_cast<const char *>(finalized.canonicalBytes().bytes().data()),
      finalized.canonicalBytes().bytes().size());
  const llvm::StringRef key = "\"architecture_fingerprint\":\"";
  const std::size_t position = tampered.find(key.str());
  require(test, position != std::string::npos,
          "canonical binding omitted architecture fingerprint");
  const std::size_t digit = position + key.size();
  tampered[digit] = tampered[digit] == '0' ? '1' : '0';
  const loom::CanonicalSemanticBytes bytes(
      std::vector<std::uint8_t>(tampered.begin(), tampered.end()));
  const auto identity =
      take(test, store.put(loom::compilerTargetBindingSchema, bytes));
  expectError(test,
              loom::importCompilerTargetBinding(
                  {loom::compilerTargetBindingSchema.identity.str(),
                   loom::compilerTargetBindingSchema.version, identity},
                  store),
              "architecture_fingerprint_mismatch");

  std::string alteredLayout(
      reinterpret_cast<const char *>(finalized.canonicalBytes().bytes().data()),
      finalized.canonicalBytes().bytes().size());
  const llvm::StringRef layoutKey = "\"data_layout\":\"";
  const std::size_t layoutPosition = alteredLayout.find(layoutKey.str());
  require(test, layoutPosition != std::string::npos,
          "canonical binding omitted DataLayout");
  const std::size_t layoutFirstByte = layoutPosition + layoutKey.size();
  alteredLayout[layoutFirstByte] =
      alteredLayout[layoutFirstByte] == 'e' ? 'E' : 'e';
  const loom::CanonicalSemanticBytes alteredLayoutBytes(
      std::vector<std::uint8_t>(alteredLayout.begin(), alteredLayout.end()));
  const auto alteredLayoutIdentity = take(
      test, store.put(loom::compilerTargetBindingSchema, alteredLayoutBytes));
  expectError(
      test,
      loom::importCompilerTargetBinding(
          {loom::compilerTargetBindingSchema.identity.str(),
           loom::compilerTargetBindingSchema.version, alteredLayoutIdentity},
          store),
      "compiler_target_reconstruction_mismatch");
}

} // namespace

int main() {
  architectureFingerprintTracksIsa();
  bindingRoundTripAndCompatibility();
  invalidAbiAndTamperedFingerprintFailClosed();
  return 0;
}
