#include "Hardware/Implementation/RepresentationFormat.h"
#include "Hardware/Implementation/RepresentationLocator.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <cstdlib>
#include <initializer_list>
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
      0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, static_cast<std::uint8_t>(kind),
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
  const auto physical =
      take(__func__, RepresentationFormatDescriptorRef::get(
                         RepresentationFormatKind::IndexedPhysical));

  require(__func__,
          hardwareRepresentationFormatRegistry.identity ==
              "loom.hardware_representation_format",
          "registry identity changed");
  require(__func__,
          hardwareRepresentationFormatRegistry.version ==
              loom::SchemaVersion{2, 1},
          "registry version changed");
  require(__func__, rtl.kind() == RepresentationFormatKind::SystemVerilogRtl,
          "RTL kind changed");
  require(__func__,
          netlist.kind() ==
              RepresentationFormatKind::StructuralVerilogGateNetlist,
          "gate-netlist kind changed");
  require(__func__,
          physical.kind() == RepresentationFormatKind::IndexedPhysical,
          "indexed-physical kind changed");

  const std::vector<std::uint8_t> rtlBytes =
      encodeRepresentationFormatDescriptorRef(rtl);
  const std::vector<std::uint8_t> netlistBytes =
      encodeRepresentationFormatDescriptorRef(netlist);
  const std::vector<std::uint8_t> physicalBytes =
      encodeRepresentationFormatDescriptorRef(physical);
  require(__func__, rtlBytes == expectedBinaryReference(rtl.kind()),
          "RTL reference bytes changed");
  require(__func__, netlistBytes == expectedBinaryReference(netlist.kind()),
          "gate-netlist reference bytes changed");
  require(__func__, physicalBytes == expectedBinaryReference(physical.kind()),
          "indexed-physical reference bytes changed");
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
  require(__func__,
          take(__func__, decodeRepresentationFormatDescriptorRef(
                             physicalBytes)) == physical,
          "indexed-physical binary reference did not round-trip");

  expectError(__func__,
              RepresentationFormatDescriptorRef::get(
                  static_cast<RepresentationFormatKind>(3)),
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
  wrongKind[54] = 3;
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
  const auto physical =
      take(__func__, RepresentationFormatDescriptorRef::get(
                         RepresentationFormatKind::IndexedPhysical));
  constexpr llvm::StringLiteral rtlJson =
      R"json({"registry":"loom.hardware_representation_format","major":2,"minor":1,"kind":0})json";
  constexpr llvm::StringLiteral netlistJson =
      R"json({"registry":"loom.hardware_representation_format","major":2,"minor":1,"kind":1})json";
  constexpr llvm::StringLiteral physicalJson =
      R"json({"registry":"loom.hardware_representation_format","major":2,"minor":1,"kind":2})json";

  require(__func__,
          serializeRepresentationFormatDescriptorRefJson(rtl) == rtlJson,
          "RTL canonical JSON changed");
  require(__func__,
          serializeRepresentationFormatDescriptorRefJson(netlist) ==
              netlistJson,
          "gate-netlist canonical JSON changed");
  require(__func__,
          serializeRepresentationFormatDescriptorRefJson(physical) ==
              physicalJson,
          "indexed-physical canonical JSON changed");
  require(__func__,
          take(__func__, parseRepresentationFormatDescriptorRefJson(rtlJson)) ==
              rtl,
          "RTL JSON reference did not round-trip");
  require(__func__,
          take(__func__, parseRepresentationFormatDescriptorRefJson(
                             netlistJson)) == netlist,
          "gate-netlist JSON reference did not round-trip");
  require(__func__,
          take(__func__, parseRepresentationFormatDescriptorRefJson(
                             physicalJson)) == physical,
          "indexed-physical JSON reference did not round-trip");

  expectError(
      __func__,
      parseRepresentationFormatDescriptorRefJson(
          R"json({"major":2,"registry":"loom.hardware_representation_format","minor":1,"kind":0})json"),
      "canonical");
  expectError(
      __func__,
      parseRepresentationFormatDescriptorRefJson(
          R"json({"registry":"loom.hardware_representation_format","major":2,"minor":1,"kind":0,"name":"sv"})json"),
      "field");
  expectError(
      __func__,
      parseRepresentationFormatDescriptorRefJson(
          R"json({"registry":"other","major":2,"minor":1,"kind":0})json"),
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
      "version");
  expectError(
      __func__,
      parseRepresentationFormatDescriptorRefJson(
          R"json({"registry":"loom.hardware_representation_format","major":2,"minor":1,"kind":3})json"),
      "kind");
  expectError(
      __func__,
      parseRepresentationFormatDescriptorRefJson(
          R"json({"registry":"loom.hardware_representation_format","major":2,"minor":1,"kind":-1})json"),
      "unsigned");
}

const RepresentationRootAdmission &
requireAdmission(llvm::StringRef test,
                 const RepresentationFormatDescriptor &descriptor,
                 RepresentationRootVariant variant,
                 std::optional<RepresentationPhysicalStage> stage) {
  for (const RepresentationRootAdmission &admission : descriptor.admittedRoots)
    if (admission.variant == variant && admission.stage == stage)
      return admission;
  fail(test, "descriptor is missing an exact root admission");
}

void requireContracts(
    llvm::StringRef test, const RepresentationRootAdmission &admission,
    std::initializer_list<RepresentationPayloadContract> expected) {
  require(test,
          admission.payloadContracts ==
              llvm::ArrayRef(expected.begin(), expected.size()),
          "root admission has the wrong payload contract");
}

void requireObjectKinds(
    llvm::StringRef test, const RepresentationRootAdmission &admission,
    std::initializer_list<RepresentationObjectKind> expected) {
  require(test,
          admission.admittedObjectKinds ==
              llvm::ArrayRef(expected.begin(), expected.size()),
          "root admission has the wrong object-kind set");
}

void staticDescriptorMetadataIsClosedWithoutCirct() {
  const RepresentationFormatDescriptorRef rtlRef =
      take(__func__, RepresentationFormatDescriptorRef::get(
                         RepresentationFormatKind::SystemVerilogRtl));
  const RepresentationFormatDescriptorRef gateRef = take(
      __func__, RepresentationFormatDescriptorRef::get(
                    RepresentationFormatKind::StructuralVerilogGateNetlist));
  const RepresentationFormatDescriptorRef physicalRef =
      take(__func__, RepresentationFormatDescriptorRef::get(
                         RepresentationFormatKind::IndexedPhysical));
  const RepresentationFormatDescriptor &rtl =
      getRepresentationFormatDescriptor(rtlRef);
  const RepresentationFormatDescriptor &gate =
      getRepresentationFormatDescriptor(gateRef);
  const RepresentationFormatDescriptor &physical =
      getRepresentationFormatDescriptor(physicalRef);

  require(__func__,
          rtl.formatRef == rtlRef && gate.formatRef == gateRef &&
              physical.formatRef == physicalRef,
          "static descriptor changed its exact format reference");
  const RepresentationRootAdmission &rtlAdmission = requireAdmission(
      __func__, rtl, RepresentationRootVariant::Rtl, std::nullopt);
  const RepresentationRootAdmission &gateAdmission = requireAdmission(
      __func__, gate, RepresentationRootVariant::GateNetlist, std::nullopt);
  requireContracts(
      __func__, rtlAdmission,
      {{PayloadRole::RtlSource, "text/x-systemverilog; charset=utf-8", 1,
        std::nullopt, RepresentationTextPolicy::Utf8LfNoNul},
       {PayloadRole::GenerationConstraint, "application/x-sdc; charset=utf-8",
        0, std::nullopt, RepresentationTextPolicy::Utf8LfNoNul},
       {PayloadRole::BlackBoxContract,
        "application/vnd.loom.black-box-contract", 0, std::nullopt,
        RepresentationTextPolicy::Opaque}});
  requireContracts(
      __func__, gateAdmission,
      {{PayloadRole::Netlist, "text/x-verilog; charset=utf-8", 1, std::nullopt,
        RepresentationTextPolicy::Utf8LfNoNul},
       {PayloadRole::GenerationConstraint, "application/x-sdc; charset=utf-8",
        0, std::nullopt, RepresentationTextPolicy::Utf8LfNoNul},
       {PayloadRole::BlackBoxContract,
        "application/vnd.loom.black-box-contract", 0, std::nullopt,
        RepresentationTextPolicy::Opaque}});

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

  requireObjectKinds(
      __func__, rtlAdmission,
      {RepresentationObjectKind::Module, RepresentationObjectKind::Instance,
       RepresentationObjectKind::Port, RepresentationObjectKind::Net,
       RepresentationObjectKind::Register, RepresentationObjectKind::Memory});
  requireObjectKinds(
      __func__, gateAdmission,
      {RepresentationObjectKind::Module, RepresentationObjectKind::Cell,
       RepresentationObjectKind::Port, RepresentationObjectKind::Pin,
       RepresentationObjectKind::Net});

  require(__func__, physical.admittedRoots.size() == 6,
          "indexed-physical descriptor has the wrong admission count");
  require(__func__, !physical.frontendSourceRole && !physical.languageProfile,
          "indexed-physical descriptor acquired an HDL frontend");

  const auto opaque = [](PayloadRole role, std::uint64_t minimum,
                         std::optional<std::uint64_t> maximum = std::nullopt) {
    return RepresentationPayloadContract{role, "application/octet-stream",
                                         minimum, maximum,
                                         RepresentationTextPolicy::Opaque};
  };
  const RepresentationPayloadContract blackBox{
      PayloadRole::BlackBoxContract, "application/vnd.loom.black-box-contract",
      0, std::nullopt, RepresentationTextPolicy::Opaque};
  const RepresentationPayloadContract index{
      PayloadRole::RepresentationIndex,
      "application/vnd.loom.physical-representation-index+json", 1,
      std::optional<std::uint64_t>(1), RepresentationTextPolicy::Utf8LfNoNul};

  const RepresentationRootAdmission &asicPlaced = requireAdmission(
      __func__, physical, RepresentationRootVariant::AsicPhysical,
      RepresentationPhysicalStage::Placed);
  const RepresentationRootAdmission &asicRouted = requireAdmission(
      __func__, physical, RepresentationRootVariant::AsicPhysical,
      RepresentationPhysicalStage::Routed);
  const RepresentationRootAdmission &asicExtracted = requireAdmission(
      __func__, physical, RepresentationRootVariant::AsicPhysical,
      RepresentationPhysicalStage::Extracted);
  const RepresentationRootAdmission &fpgaPlaced = requireAdmission(
      __func__, physical, RepresentationRootVariant::FpgaPhysical,
      RepresentationPhysicalStage::Placed);
  const RepresentationRootAdmission &fpgaRouted = requireAdmission(
      __func__, physical, RepresentationRootVariant::FpgaPhysical,
      RepresentationPhysicalStage::Routed);
  const RepresentationRootAdmission &fpgaImage = requireAdmission(
      __func__, physical, RepresentationRootVariant::FpgaImage, std::nullopt);

  requireContracts(__func__, asicPlaced,
                   {opaque(PayloadRole::PhysicalDatabase, 1),
                    opaque(PayloadRole::GenerationConstraint, 0), blackBox,
                    index});
  requireContracts(__func__, asicRouted,
                   {opaque(PayloadRole::PhysicalDatabase, 1),
                    opaque(PayloadRole::LayoutStream, 0),
                    opaque(PayloadRole::GenerationConstraint, 0), blackBox,
                    index});
  requireContracts(
      __func__, asicExtracted,
      {opaque(PayloadRole::PhysicalDatabase, 1),
       opaque(PayloadRole::Parasitics, 1), opaque(PayloadRole::LayoutStream, 0),
       opaque(PayloadRole::GenerationConstraint, 0), blackBox, index});
  for (const RepresentationRootAdmission *admission :
       {&fpgaPlaced, &fpgaRouted})
    requireContracts(__func__, *admission,
                     {opaque(PayloadRole::PhysicalDatabase, 1),
                      opaque(PayloadRole::GenerationConstraint, 0), blackBox,
                      index});
  requireContracts(
      __func__, fpgaImage,
      {opaque(PayloadRole::DeviceImage, 1, std::optional<std::uint64_t>(1)),
       index});

  for (const RepresentationRootAdmission *admission :
       {&asicPlaced, &asicRouted, &asicExtracted}) {
    require(__func__,
            admission->exactRootKind ==
                RepresentationObjectKind::PhysicalObject,
            "ASIC physical admission has the wrong root kind");
    requireObjectKinds(
        __func__, *admission,
        {RepresentationObjectKind::Module, RepresentationObjectKind::Instance,
         RepresentationObjectKind::Port, RepresentationObjectKind::Net,
         RepresentationObjectKind::Register, RepresentationObjectKind::Memory,
         RepresentationObjectKind::Cell, RepresentationObjectKind::Pin,
         RepresentationObjectKind::PhysicalObject});
  }
  for (const RepresentationRootAdmission *admission :
       {&fpgaPlaced, &fpgaRouted, &fpgaImage}) {
    require(__func__,
            admission->exactRootKind ==
                RepresentationObjectKind::DeviceResource,
            "FPGA admission has the wrong root kind");
    requireObjectKinds(
        __func__, *admission,
        {RepresentationObjectKind::Module, RepresentationObjectKind::Instance,
         RepresentationObjectKind::Port, RepresentationObjectKind::Net,
         RepresentationObjectKind::Register, RepresentationObjectKind::Memory,
         RepresentationObjectKind::Cell, RepresentationObjectKind::Pin,
         RepresentationObjectKind::DeviceResource});
  }
}

} // namespace

int main() {
  exactBinaryCodecIsClosed();
  exactJsonCodecIsClosed();
  staticDescriptorMetadataIsClosedWithoutCirct();
  return EXIT_SUCCESS;
}
