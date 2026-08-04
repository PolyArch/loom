#include "ADG/Builtin.h"
#include "Common/ArtifactStore.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Hardware/RTL/OperationLeaf.h"

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <optional>
#include <string>
#include <utility>

namespace {

using loom::fabric::FabricPortDirection;
using loom::fabric::FinalizedFabricRoot;
using loom::fabric::ResolvedFabricOpCapabilityView;
using loom::hardware::rtl::TransparentLoopOperationLeafStateLayout;

[[noreturn]] void fail(llvm::StringRef test, const std::string &message) {
  llvm::errs() << test << ": " << message << '\n';
  std::exit(EXIT_FAILURE);
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

template <typename T>
void expectError(llvm::StringRef test, llvm::Expected<T> value,
                 llvm::StringRef expected) {
  if (value)
    fail(test, "accepted an invalid loop state layout");
  const std::string message = llvm::toString(value.takeError());
  require(test, llvm::StringRef(message).contains(expected), message);
}

const ResolvedFabricOpCapabilityView *
findCapability(const FinalizedFabricRoot &fabric,
               ::fabric::ImplementationFamilyId family) {
  for (const auto occurrence : fabric.view().fuOccurrences()) {
    const auto definition = fabric.view().fuTemplateOf(occurrence);
    if (!definition)
      continue;
    const auto capabilities =
        fabric.view().resolvedFabricOpCapabilities(*definition);
    const auto found = llvm::find_if(capabilities, [&](const auto &candidate) {
      return candidate.implementationFamily == family;
    });
    if (found != capabilities.end())
      return &*found;
  }
  return nullptr;
}

unsigned payloadWidth(const ResolvedFabricOpCapabilityView &capability,
                      FabricPortDirection direction, std::uint64_t ordinal) {
  const auto found =
      llvm::find_if(capability.physicalPorts, [&](const auto &port) {
        return port.reference.direction == direction &&
               port.reference.ordinal == ordinal;
      });
  return found == capability.physicalPorts.end() ? 0 : found->payloadWidthBits;
}

void setPayloadWidth(ResolvedFabricOpCapabilityView &capability,
                     FabricPortDirection direction, std::uint64_t ordinal,
                     unsigned width) {
  const auto found =
      llvm::find_if(capability.physicalPorts, [&](const auto &port) {
        return port.reference.direction == direction &&
               port.reference.ordinal == ordinal;
      });
  if (found == capability.physicalPorts.end())
    fail("setPayloadWidth", "fixture is missing a physical port");
  found->payloadWidthBits = width;
}

TransparentLoopOperationLeafStateLayout
requireLayout(llvm::StringRef test,
              const ResolvedFabricOpCapabilityView &capability) {
  std::optional<TransparentLoopOperationLeafStateLayout> layout = take(
      test, loom::hardware::rtl::deriveTransparentLoopOperationLeafStateLayout(
                capability));
  require(test, layout.has_value(), "stateful loop family has no layout");
  require(test,
          layout->resetValue().getBitWidth() == layout->encodedBitCount() &&
              layout->resetValue().isZero(),
          "loop state layout has a noncanonical reset value");
  return *layout;
}

void builtinLayoutsUseExactFabricPorts(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root);
  loom::ArtifactStore store(root.string());
  auto design = take(test, loom::adg::buildBuiltinTarget(
                               store, loom::adg::BuiltinTargetPreset::Small));
  require(test, design.roots().size() == 1,
          "Small builtin did not publish one System root");
  const auto dependencies = design.roots().front().directDependencies();
  require(test, dependencies.size() == 1,
          "Small builtin did not publish one Module dependency");
  FinalizedFabricRoot module =
      take(test, loom::fabric::importEntireFabricRoot(dependencies.front().root,
                                                      store));

  const auto *carry =
      findCapability(module, ::fabric::ImplementationFamilyId::LoopCarry);
  const auto *invariant =
      findCapability(module, ::fabric::ImplementationFamilyId::LoopInvariant);
  const auto *gate =
      findCapability(module, ::fabric::ImplementationFamilyId::LoopGate);
  const auto *multiply = findCapability(
      module, ::fabric::ImplementationFamilyId::ScalarIntegerMultiply);
  require(test, carry && invariant && gate && multiply,
          "Small builtin is missing a required capability");

  require(test,
          payloadWidth(*invariant, FabricPortDirection::Input, 0) == 1 &&
              payloadWidth(*invariant, FabricPortDirection::Input, 1) == 128 &&
              payloadWidth(*invariant, FabricPortDirection::Output, 0) == 128,
          "Small invariant physical shape changed");
  require(test,
          payloadWidth(*gate, FabricPortDirection::Input, 0) == 1 &&
              payloadWidth(*gate, FabricPortDirection::Input, 1) == 128 &&
              payloadWidth(*gate, FabricPortDirection::Output, 0) == 1 &&
              payloadWidth(*gate, FabricPortDirection::Output, 1) == 128,
          "Small gate physical shape changed");

  const auto carryLayout = requireLayout(test, *carry);
  const auto invariantLayout = requireLayout(test, *invariant);
  const auto gateLayout = requireLayout(test, *gate);
  require(test,
          TransparentLoopOperationLeafStateLayout::modeBit == 0 &&
              TransparentLoopOperationLeafStateLayout::invariantPayloadOffset ==
                  1 &&
              carryLayout.payloadWidthBits == 0 &&
              carryLayout.encodedBitCount() == 1 &&
              invariantLayout.payloadWidthBits == 128 &&
              invariantLayout.encodedBitCount() == 129 &&
              gateLayout.payloadWidthBits == 0 &&
              gateLayout.encodedBitCount() == 1,
          "Small loop state layouts changed");
  auto stateless = take(
      test, loom::hardware::rtl::deriveTransparentLoopOperationLeafStateLayout(
                *multiply));
  require(test, !stateless,
          "stateless operation acquired a selected-context layout");
}

void invariantPayloadGeometryIsMechanical(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root);
  loom::ArtifactStore store(root.string());
  auto design = take(test, loom::adg::buildBuiltinTarget(
                               store, loom::adg::BuiltinTargetPreset::Small));
  FinalizedFabricRoot module =
      take(test, loom::fabric::importEntireFabricRoot(
                     design.roots().front().directDependencies().front().root,
                     store));
  const auto *canonical =
      findCapability(module, ::fabric::ImplementationFamilyId::LoopInvariant);
  require(test, canonical != nullptr, "Small invariant capability is absent");

  ResolvedFabricOpCapabilityView mixed = *canonical;
  setPayloadWidth(mixed, FabricPortDirection::Input, 1, 64);
  setPayloadWidth(mixed, FabricPortDirection::Output, 0, 96);
  const auto mixedLayout = requireLayout(test, mixed);
  require(test,
          mixedLayout.payloadWidthBits == 64 &&
              mixedLayout.encodedBitCount() == 65,
          "mixed invariant layout did not use the shared payload capacity");

  ResolvedFabricOpCapabilityView zero = *canonical;
  setPayloadWidth(zero, FabricPortDirection::Input, 1, 0);
  setPayloadWidth(zero, FabricPortDirection::Output, 0, 0);
  const auto zeroLayout = requireLayout(test, zero);
  require(test,
          zeroLayout.payloadWidthBits == 0 && zeroLayout.encodedBitCount() == 1,
          "zero-payload invariant did not retain only its mode bit");

  ResolvedFabricOpCapabilityView missing = *canonical;
  llvm::erase_if(missing.physicalPorts, [](const auto &port) {
    return port.reference.direction == FabricPortDirection::Input &&
           port.reference.ordinal == 1;
  });
  expectError(
      test,
      loom::hardware::rtl::deriveTransparentLoopOperationLeafStateLayout(
          missing),
      "payload port");

  ResolvedFabricOpCapabilityView oversized = *canonical;
  setPayloadWidth(oversized, FabricPortDirection::Input, 1,
                  mlir::IntegerType::kMaxWidth);
  setPayloadWidth(oversized, FabricPortDirection::Output, 0,
                  mlir::IntegerType::kMaxWidth);
  expectError(
      test,
      loom::hardware::rtl::deriveTransparentLoopOperationLeafStateLayout(
          oversized),
      "CIRCT integer limit");
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2)
    fail("main", "expected one temporary directory argument");
  const std::filesystem::path root(argv[1]);
  builtinLayoutsUseExactFabricPorts(root / "builtin");
  invariantPayloadGeometryIsMechanical(root / "geometry");
  return 0;
}
