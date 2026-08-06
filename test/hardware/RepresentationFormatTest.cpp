#include "Hardware/Implementation/RepresentationFormat.h"
#include "Hardware/Implementation/RepresentationLocator.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <cstdlib>
#include <optional>
#include <string>
#include <utility>
#include <vector>

using namespace loom::hardware;

namespace {

[[noreturn]] void fail(llvm::StringRef test, const std::string &message) {
  llvm::errs() << test << ": " << message << '\n';
  std::exit(EXIT_FAILURE);
}

void require(llvm::StringRef test, bool condition, const std::string &message) {
  if (!condition)
    fail(test, message);
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
    fail(test, "accepted invalid representation format reference");
  const std::string message = llvm::toString(value.takeError());
  require(test, llvm::StringRef(message).contains(expected), message);
}

std::vector<std::uint8_t>
expectedBinaryReference(RepresentationFormatKind kind) {
  constexpr llvm::StringLiteral identity =
      "loom.hardware_representation_format";
  std::vector<std::uint8_t> expected{0, 0, 0, 0, 0, 0, 0, 35};
  expected.insert(expected.end(), identity.bytes_begin(), identity.bytes_end());
  const std::vector<std::uint8_t> suffix{
      0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, static_cast<std::uint8_t>(kind),
  };
  expected.insert(expected.end(), suffix.begin(), suffix.end());
  return expected;
}

void exactBinaryCodecIsClosed() {
  const auto rtl =
      take(__func__, RepresentationFormatDescriptorRef::get(
                         RepresentationFormatKind::SystemVerilogRtl));
  const auto netlist = take(
      __func__, RepresentationFormatDescriptorRef::get(
                    RepresentationFormatKind::StructuralVerilogGateNetlist));

  require(__func__,
          hardwareRepresentationFormatRegistry.identity ==
              "loom.hardware_representation_format",
          "registry identity changed");
  require(__func__,
          hardwareRepresentationFormatRegistry.version ==
              loom::SchemaVersion{2, 0},
          "registry version changed");
  require(__func__, rtl.kind() == RepresentationFormatKind::SystemVerilogRtl,
          "RTL kind changed");
  require(__func__,
          netlist.kind() ==
              RepresentationFormatKind::StructuralVerilogGateNetlist,
          "gate-netlist kind changed");

  const std::vector<std::uint8_t> rtlBytes =
      encodeRepresentationFormatDescriptorRef(rtl);
  const std::vector<std::uint8_t> netlistBytes =
      encodeRepresentationFormatDescriptorRef(netlist);
  require(__func__, rtlBytes == expectedBinaryReference(rtl.kind()),
          "RTL reference bytes changed");
  require(__func__, netlistBytes == expectedBinaryReference(netlist.kind()),
          "gate-netlist reference bytes changed");
  require(__func__, rtlBytes.size() == 55,
          "reference framing has the wrong size");
  require(__func__,
          take(__func__, decodeRepresentationFormatDescriptorRef(rtlBytes)) ==
              rtl,
          "RTL binary reference did not round-trip");
  require(__func__,
          take(__func__, decodeRepresentationFormatDescriptorRef(
                             netlistBytes)) == netlist,
          "gate-netlist binary reference did not round-trip");

  expectError(__func__,
              RepresentationFormatDescriptorRef::get(
                  static_cast<RepresentationFormatKind>(2)),
              "kind");

  for (std::size_t size = 0; size < rtlBytes.size(); ++size)
    expectError(__func__,
                decodeRepresentationFormatDescriptorRef(
                    llvm::ArrayRef(rtlBytes).take_front(size)),
                "truncated");

  std::vector<std::uint8_t> trailing = rtlBytes;
  trailing.push_back(0);
  expectError(__func__, decodeRepresentationFormatDescriptorRef(trailing),
              "trailing");

  std::vector<std::uint8_t> wrongRegistry = rtlBytes;
  wrongRegistry[8] = 'L';
  expectError(__func__, decodeRepresentationFormatDescriptorRef(wrongRegistry),
              "registry");

  std::vector<std::uint8_t> wrongVersion = rtlBytes;
  wrongVersion[46] = 1;
  expectError(__func__, decodeRepresentationFormatDescriptorRef(wrongVersion),
              "version");

  std::vector<std::uint8_t> wrongKind = rtlBytes;
  wrongKind[54] = 2;
  expectError(__func__, decodeRepresentationFormatDescriptorRef(wrongKind),
              "kind");
}

void exactJsonCodecIsClosed() {
  const auto rtl =
      take(__func__, RepresentationFormatDescriptorRef::get(
                         RepresentationFormatKind::SystemVerilogRtl));
  const auto netlist = take(
      __func__, RepresentationFormatDescriptorRef::get(
                    RepresentationFormatKind::StructuralVerilogGateNetlist));
  constexpr llvm::StringLiteral rtlJson =
      R"json({"registry":"loom.hardware_representation_format","major":2,"minor":0,"kind":0})json";
  constexpr llvm::StringLiteral netlistJson =
      R"json({"registry":"loom.hardware_representation_format","major":2,"minor":0,"kind":1})json";

  require(__func__,
          serializeRepresentationFormatDescriptorRefJson(rtl) == rtlJson,
          "RTL canonical JSON changed");
  require(__func__,
          serializeRepresentationFormatDescriptorRefJson(netlist) ==
              netlistJson,
          "gate-netlist canonical JSON changed");
  require(__func__,
          take(__func__, parseRepresentationFormatDescriptorRefJson(rtlJson)) ==
              rtl,
          "RTL JSON reference did not round-trip");
  require(__func__,
          take(__func__, parseRepresentationFormatDescriptorRefJson(
                             netlistJson)) == netlist,
          "gate-netlist JSON reference did not round-trip");

  expectError(
      __func__,
      parseRepresentationFormatDescriptorRefJson(
          R"json({"major":2,"registry":"loom.hardware_representation_format","minor":0,"kind":0})json"),
      "canonical");
  expectError(
      __func__,
      parseRepresentationFormatDescriptorRefJson(
          R"json({"registry":"loom.hardware_representation_format","major":2,"minor":0,"kind":0,"name":"sv"})json"),
      "field");
  expectError(
      __func__,
      parseRepresentationFormatDescriptorRefJson(
          R"json({"registry":"other","major":2,"minor":0,"kind":0})json"),
      "registry");
  expectError(
      __func__,
      parseRepresentationFormatDescriptorRefJson(
          R"json({"registry":"loom.hardware_representation_format","major":1,"minor":0,"kind":0})json"),
      "version");
  expectError(
      __func__,
      parseRepresentationFormatDescriptorRefJson(
          R"json({"registry":"loom.hardware_representation_format","major":2,"minor":0,"kind":2})json"),
      "kind");
  expectError(
      __func__,
      parseRepresentationFormatDescriptorRefJson(
          R"json({"registry":"loom.hardware_representation_format","major":2,"minor":0,"kind":-1})json"),
      "unsigned");
}

void staticDescriptorMetadataIsClosedWithoutCirct() {
  const RepresentationFormatDescriptorRef rtlRef =
      take(__func__, RepresentationFormatDescriptorRef::get(
                         RepresentationFormatKind::SystemVerilogRtl));
  const RepresentationFormatDescriptorRef gateRef = take(
      __func__, RepresentationFormatDescriptorRef::get(
                    RepresentationFormatKind::StructuralVerilogGateNetlist));
  const RepresentationFormatDescriptor &rtl =
      getRepresentationFormatDescriptor(rtlRef);
  const RepresentationFormatDescriptor &gate =
      getRepresentationFormatDescriptor(gateRef);

  require(__func__, rtl.formatRef == rtlRef && gate.formatRef == gateRef,
          "static descriptor changed its exact format reference");
  require(__func__,
          rtl.exactRootKind == RepresentationObjectKind::Module &&
              gate.exactRootKind == RepresentationObjectKind::Module,
          "initial HDL descriptor changed its exact root kind");
  require(__func__,
          rtl.payloadContracts.size() == 3 && gate.payloadContracts.size() == 3,
          "initial HDL descriptor has the wrong role closure");
  require(__func__,
          rtl.payloadContracts[0] ==
              RepresentationPayloadContract{
                  PayloadRole::RtlSource, "text/x-systemverilog; charset=utf-8",
                  1, std::nullopt, RepresentationTextPolicy::Utf8LfNoNul},
          "RTL source contract changed");
  require(__func__,
          gate.payloadContracts[0] ==
              RepresentationPayloadContract{
                  PayloadRole::Netlist, "text/x-verilog; charset=utf-8", 1,
                  std::nullopt, RepresentationTextPolicy::Utf8LfNoNul},
          "gate-netlist source contract changed");
  for (llvm::ArrayRef<RepresentationPayloadContract> contracts :
       {rtl.payloadContracts, gate.payloadContracts}) {
    require(__func__,
            contracts[1] ==
                RepresentationPayloadContract{
                    PayloadRole::GenerationConstraint,
                    "application/x-sdc; charset=utf-8", 0, std::nullopt,
                    RepresentationTextPolicy::Utf8LfNoNul},
            "generation-constraint contract changed");
    require(__func__,
            contracts[2] ==
                RepresentationPayloadContract{
                    PayloadRole::BlackBoxContract,
                    "application/vnd.loom.black-box-contract", 0, std::nullopt,
                    RepresentationTextPolicy::Opaque},
            "black-box contract changed");
  }

  require(__func__,
          rtl.frontendSourceRole == std::optional(PayloadRole::RtlSource) &&
              gate.frontendSourceRole == std::optional(PayloadRole::Netlist),
          "descriptor source-role ownership changed");
  require(__func__,
          rtl.languageProfile ==
                  std::optional(RepresentationLanguageProfile::Ieee1800_2017) &&
              gate.languageProfile ==
                  std::optional(RepresentationLanguageProfile::Ieee1364_2005),
          "descriptor language profiles changed");

  const std::vector<RepresentationObjectKind> expectedRtlKinds{
      RepresentationObjectKind::Module,   RepresentationObjectKind::Instance,
      RepresentationObjectKind::Port,     RepresentationObjectKind::Net,
      RepresentationObjectKind::Register, RepresentationObjectKind::Memory};
  const std::vector<RepresentationObjectKind> expectedGateKinds{
      RepresentationObjectKind::Module, RepresentationObjectKind::Cell,
      RepresentationObjectKind::Port, RepresentationObjectKind::Pin,
      RepresentationObjectKind::Net};
  require(__func__, rtl.admittedObjectKinds == llvm::ArrayRef(expectedRtlKinds),
          "RTL admitted object-kind set changed");
  require(__func__,
          gate.admittedObjectKinds == llvm::ArrayRef(expectedGateKinds),
          "gate admitted object-kind set changed");
}

} // namespace

int main() {
  exactBinaryCodecIsClosed();
  exactJsonCodecIsClosed();
  staticDescriptorMetadataIsClosedWithoutCirct();
  return EXIT_SUCCESS;
}
