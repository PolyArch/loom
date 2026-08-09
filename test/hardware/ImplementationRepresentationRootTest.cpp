#include "Hardware/Implementation/ImplementationRepresentationRoot.h"

#include "Common/BlobDigest.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <optional>
#include <string>
#include <utility>
#include <vector>

using namespace loom;
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
    fail(test, "accepted an invalid representation root");
  const std::string message = llvm::toString(value.takeError());
  require(test, llvm::StringRef(message).contains(expected), message);
}

BlobDigest digest(llvm::StringRef contents) {
  return computeBlobDigest(llvm::ArrayRef<std::uint8_t>(contents.bytes_begin(),
                                                        contents.bytes_end()));
}

RepresentationFormatDescriptorRef rtlFormat(llvm::StringRef test) {
  return take(test, RepresentationFormatDescriptorRef::get(
                        RepresentationFormatKind::SystemVerilogRtl));
}

RepresentationFormatDescriptorRef gateFormat(llvm::StringRef test) {
  return take(test,
              RepresentationFormatDescriptorRef::get(
                  RepresentationFormatKind::StructuralVerilogGateNetlist));
}

RepresentationFormatDescriptorRef physicalFormat(llvm::StringRef test) {
  return take(test, RepresentationFormatDescriptorRef::get(
                        RepresentationFormatKind::IndexedPhysical));
}

std::vector<ImplementationPayload> rtlPayloads() {
  return {{PayloadRole::RtlSource, "rtl/top.sv", digest("rtl")},
          {PayloadRole::BlackBoxContract, "ip/pll.bb", digest("ip")}};
}

std::vector<ImplementationPayload>
physicalPayloads(std::initializer_list<PayloadRole> roles) {
  std::vector<ImplementationPayload> payloads;
  for (PayloadRole role : roles)
    payloads.push_back(
        {role,
         role == PayloadRole::RepresentationIndex
             ? "index/physical.json"
             : ("payload/" + std::to_string(static_cast<std::uint32_t>(role))),
         digest(llvm::Twine(static_cast<std::uint32_t>(role)).str())});
  return take(__func__, canonicalizeImplementationPayloadCatalog(payloads));
}

ImplementationRepresentationRoot
makeRoot(RepresentationRootVariant variant,
         std::optional<RepresentationPhysicalStage> stage,
         RepresentationFormatDescriptorRef format,
         RepresentationObjectKind topKind, llvm::StringRef test) {
  return take(test,
              createImplementationRepresentationRoot(
                  variant, stage, format, {topKind, "top"}, rtlPayloads()));
}

void appendBytes(std::vector<std::uint8_t> &target,
                 const std::vector<std::uint8_t> &source) {
  target.insert(target.end(), source.begin(), source.end());
}

std::vector<std::uint8_t>
expectedRootBytes(RepresentationRootVariant variant,
                  std::optional<RepresentationPhysicalStage> stage,
                  RepresentationFormatDescriptorRef format,
                  const RepresentationLocator &top,
                  const std::vector<ImplementationPayload> &payloads) {
  std::vector<std::uint8_t> expected;
  const std::uint32_t tag = static_cast<std::uint32_t>(variant);
  for (unsigned shift = 24; shift != 0; shift -= 8)
    expected.push_back(static_cast<std::uint8_t>(tag >> shift));
  expected.push_back(static_cast<std::uint8_t>(tag));
  if (stage) {
    const std::uint32_t stageTag = static_cast<std::uint32_t>(*stage);
    for (unsigned shift = 24; shift != 0; shift -= 8)
      expected.push_back(static_cast<std::uint8_t>(stageTag >> shift));
    expected.push_back(static_cast<std::uint8_t>(stageTag));
  }
  appendBytes(expected, encodeRepresentationFormatDescriptorRef(format));
  appendBytes(expected,
              take("expectedRootBytes", encodeRepresentationLocator(top)));
  expected.insert(expected.end(), 8 - 1, 0);
  expected.push_back(static_cast<std::uint8_t>(payloads.size()));
  for (const ImplementationPayload &payload : payloads)
    appendBytes(expected, take("expectedRootBytes",
                               encodeImplementationPayload(payload)));
  return expected;
}

void closedVariantsRoundTripExactly() {
  struct Case {
    RepresentationRootVariant variant;
    std::optional<RepresentationPhysicalStage> stage;
    RepresentationObjectKind topKind;
    llvm::StringRef variantSpelling;
    std::optional<llvm::StringRef> stageSpelling;
  };
  const std::vector<Case> cases{
      {RepresentationRootVariant::Rtl, std::nullopt,
       RepresentationObjectKind::Module, "Rtl", std::nullopt},
      {RepresentationRootVariant::GateNetlist, std::nullopt,
       RepresentationObjectKind::Module, "GateNetlist", std::nullopt},
      {RepresentationRootVariant::AsicPhysical,
       RepresentationPhysicalStage::Placed,
       RepresentationObjectKind::PhysicalObject, "AsicPhysical", "Placed"},
      {RepresentationRootVariant::AsicPhysical,
       RepresentationPhysicalStage::Routed,
       RepresentationObjectKind::PhysicalObject, "AsicPhysical", "Routed"},
      {RepresentationRootVariant::AsicPhysical,
       RepresentationPhysicalStage::Extracted,
       RepresentationObjectKind::PhysicalObject, "AsicPhysical", "Extracted"},
      {RepresentationRootVariant::FpgaPhysical,
       RepresentationPhysicalStage::Placed,
       RepresentationObjectKind::DeviceResource, "FpgaPhysical", "Placed"},
      {RepresentationRootVariant::FpgaPhysical,
       RepresentationPhysicalStage::Routed,
       RepresentationObjectKind::DeviceResource, "FpgaPhysical", "Routed"},
      {RepresentationRootVariant::FpgaImage, std::nullopt,
       RepresentationObjectKind::DeviceResource, "FpgaImage", std::nullopt},
  };
  for (const Case &entry : cases) {
    const ImplementationRepresentationRoot root =
        makeRoot(entry.variant, entry.stage, rtlFormat(__func__), entry.topKind,
                 __func__);
    require(__func__,
            root.variant == entry.variant && root.stage == entry.stage,
            "root lost its variant or stage");

    const std::vector<std::uint8_t> expected =
        expectedRootBytes(entry.variant, entry.stage, rtlFormat(__func__),
                          {entry.topKind, "top"}, rtlPayloads());
    const std::vector<std::uint8_t> bytes =
        take(__func__, encodeImplementationRepresentationRoot(root));
    require(__func__, bytes == expected, "root binary framing changed");
    require(__func__,
            take(__func__, decodeImplementationRepresentationRoot(bytes)) ==
                root,
            "root binary did not round-trip");

    std::string expectedJson = "{\"variant\":\"";
    expectedJson += entry.variantSpelling.str();
    expectedJson += "\",";
    if (entry.stageSpelling) {
      expectedJson += "\"stage\":\"";
      expectedJson += entry.stageSpelling->str();
      expectedJson += "\",";
    }
    expectedJson += "\"format_ref\":";
    expectedJson +=
        serializeRepresentationFormatDescriptorRefJson(rtlFormat(__func__));
    expectedJson += ",\"top\":";
    expectedJson += take(
        __func__, serializeRepresentationLocatorJson({entry.topKind, "top"}));
    expectedJson += ",\"payloads\":[";
    bool first = true;
    for (const ImplementationPayload &payload : rtlPayloads()) {
      if (!first)
        expectedJson += ",";
      first = false;
      expectedJson +=
          take(__func__, serializeImplementationPayloadJson(payload));
    }
    expectedJson += "]}";
    const std::string json =
        take(__func__, serializeImplementationRepresentationRootJson(root));
    require(__func__, json == expectedJson, "root canonical JSON changed");
    require(__func__,
            take(__func__, parseImplementationRepresentationRootJson(json)) ==
                root,
            "root JSON did not round-trip");
  }
}

void binaryFramingIsStrict() {
  const ImplementationRepresentationRoot root =
      makeRoot(RepresentationRootVariant::Rtl, std::nullopt,
               rtlFormat(__func__), RepresentationObjectKind::Module, __func__);
  const std::vector<std::uint8_t> bytes =
      take(__func__, encodeImplementationRepresentationRoot(root));
  for (std::size_t size = 0; size < bytes.size(); ++size)
    expectError(__func__,
                decodeImplementationRepresentationRoot(
                    llvm::ArrayRef(bytes).take_front(size)),
                "truncated");
  std::vector<std::uint8_t> trailing = bytes;
  trailing.push_back(0);
  expectError(__func__, decodeImplementationRepresentationRoot(trailing),
              "trailing");

  std::vector<std::uint8_t> unknownVariant = bytes;
  unknownVariant[3] = 5;
  expectError(__func__, decodeImplementationRepresentationRoot(unknownVariant),
              "variant");

  // An Rtl root carrying a stage tag is not decodable as the selected
  // variant; the stage tag is misparsed as the format-reference length.
  std::vector<std::uint8_t> stageOnRtl = bytes;
  stageOnRtl.insert(stageOnRtl.begin() + 4, {0, 0, 0, 0});
  expectError(__func__, decodeImplementationRepresentationRoot(stageOnRtl), "");

  std::vector<std::uint8_t> wrongStage = bytes;
  std::vector<std::uint8_t> physical = take(
      __func__, encodeImplementationRepresentationRoot(makeRoot(
                    RepresentationRootVariant::AsicPhysical,
                    RepresentationPhysicalStage::Placed, rtlFormat(__func__),
                    RepresentationObjectKind::PhysicalObject, __func__)));
  wrongStage = physical;
  wrongStage[7] = 3;
  expectError(__func__, decodeImplementationRepresentationRoot(wrongStage),
              "stage");
  std::vector<std::uint8_t> fpgaExtracted = take(
      __func__, encodeImplementationRepresentationRoot(makeRoot(
                    RepresentationRootVariant::FpgaPhysical,
                    RepresentationPhysicalStage::Routed, rtlFormat(__func__),
                    RepresentationObjectKind::DeviceResource, __func__)));
  fpgaExtracted[7] = 2;
  expectError(__func__, decodeImplementationRepresentationRoot(fpgaExtracted),
              "stage");
}

void variantStageLegalityIsClosedAtAuthoring() {
  expectError(__func__,
              createImplementationRepresentationRoot(
                  RepresentationRootVariant::Rtl,
                  RepresentationPhysicalStage::Placed, rtlFormat(__func__),
                  {RepresentationObjectKind::Module, "top"}, rtlPayloads()),
              "stage");
  expectError(__func__,
              createImplementationRepresentationRoot(
                  RepresentationRootVariant::FpgaImage,
                  RepresentationPhysicalStage::Placed, rtlFormat(__func__),
                  {RepresentationObjectKind::DeviceResource, "top"},
                  rtlPayloads()),
              "stage");
  expectError(__func__,
              createImplementationRepresentationRoot(
                  RepresentationRootVariant::AsicPhysical, std::nullopt,
                  rtlFormat(__func__),
                  {RepresentationObjectKind::PhysicalObject, "top"},
                  rtlPayloads()),
              "stage");
  expectError(__func__,
              createImplementationRepresentationRoot(
                  RepresentationRootVariant::FpgaPhysical,
                  RepresentationPhysicalStage::Extracted, rtlFormat(__func__),
                  {RepresentationObjectKind::DeviceResource, "top"},
                  rtlPayloads()),
              "stage");
}

void topKindMatchesVariant() {
  expectError(
      __func__,
      createImplementationRepresentationRoot(
          RepresentationRootVariant::Rtl, std::nullopt, rtlFormat(__func__),
          {RepresentationObjectKind::PhysicalObject, "top"}, rtlPayloads()),
      "top");
  expectError(__func__,
              createImplementationRepresentationRoot(
                  RepresentationRootVariant::GateNetlist, std::nullopt,
                  gateFormat(__func__),
                  {RepresentationObjectKind::DeviceResource, "top"},
                  rtlPayloads()),
              "top");
  expectError(__func__,
              createImplementationRepresentationRoot(
                  RepresentationRootVariant::AsicPhysical,
                  RepresentationPhysicalStage::Placed, rtlFormat(__func__),
                  {RepresentationObjectKind::Module, "top"}, rtlPayloads()),
              "top");
  expectError(__func__,
              createImplementationRepresentationRoot(
                  RepresentationRootVariant::FpgaImage, std::nullopt,
                  rtlFormat(__func__),
                  {RepresentationObjectKind::Module, "top"}, rtlPayloads()),
              "top");
}

void payloadCatalogIsCanonicalAndUnique() {
  expectError(__func__,
              createImplementationRepresentationRoot(
                  RepresentationRootVariant::Rtl, std::nullopt,
                  rtlFormat(__func__),
                  {RepresentationObjectKind::Module, "top"}, {}),
              "nonempty");
  std::vector<ImplementationPayload> duplicates = rtlPayloads();
  duplicates.push_back(duplicates.front());
  expectError(__func__,
              createImplementationRepresentationRoot(
                  RepresentationRootVariant::Rtl, std::nullopt,
                  rtlFormat(__func__),
                  {RepresentationObjectKind::Module, "top"}, duplicates),
              "duplicate");

  std::vector<ImplementationPayload> reversed = rtlPayloads();
  std::reverse(reversed.begin(), reversed.end());
  const ImplementationRepresentationRoot authored =
      take(__func__, createImplementationRepresentationRoot(
                         RepresentationRootVariant::Rtl, std::nullopt,
                         rtlFormat(__func__),
                         {RepresentationObjectKind::Module, "top"}, reversed));
  const ImplementationRepresentationRoot canonical =
      makeRoot(RepresentationRootVariant::Rtl, std::nullopt,
               rtlFormat(__func__), RepresentationObjectKind::Module, __func__);
  require(__func__, authored == canonical,
          "authoring order changed the canonical root");
  require(__func__,
          take(__func__, encodeImplementationRepresentationRoot(authored)) ==
              take(__func__, encodeImplementationRepresentationRoot(canonical)),
          "authoring order changed the canonical bytes");

  const std::vector<std::uint8_t> canonicalBytes =
      take(__func__, encodeImplementationRepresentationRoot(canonical));
  // Reorder the two payload records inside the framed array; both payloads
  // are fixed-width here only in spirit, so swap by re-encoding.
  std::vector<std::uint8_t> reordered;
  const std::vector<std::uint8_t> first =
      take(__func__, encodeImplementationPayload(rtlPayloads().front()));
  const std::vector<std::uint8_t> second =
      take(__func__, encodeImplementationPayload(rtlPayloads().back()));
  const std::size_t header =
      canonicalBytes.size() - first.size() - second.size();
  reordered.insert(reordered.end(), canonicalBytes.begin(),
                   canonicalBytes.begin() + header);
  appendBytes(reordered, second);
  appendBytes(reordered, first);
  expectError(__func__, decodeImplementationRepresentationRoot(reordered),
              "canonical");
}

bool admissionFails(llvm::StringRef test,
                    const RepresentationFormatDescriptor &descriptor,
                    const ImplementationRepresentationRoot &root) {
  llvm::Error error = validateRepresentationRootAdmission(descriptor, root);
  if (!error)
    return false;
  llvm::consumeError(std::move(error));
  return true;
}

void descriptorAdmissionIsDataDriven() {
  const RepresentationFormatDescriptor &rtl =
      getRepresentationFormatDescriptor(rtlFormat(__func__));
  const RepresentationFormatDescriptor &gate =
      getRepresentationFormatDescriptor(gateFormat(__func__));
  const RepresentationFormatDescriptor &physical =
      getRepresentationFormatDescriptor(physicalFormat(__func__));
  require(__func__,
          admitsRepresentationRoot(rtl, RepresentationRootVariant::Rtl,
                                   std::nullopt),
          "RTL descriptor lost its stageless Rtl admission");
  require(__func__,
          !admitsRepresentationRoot(rtl, RepresentationRootVariant::Rtl,
                                    RepresentationPhysicalStage::Placed),
          "RTL descriptor admitted a staged Rtl root");
  require(__func__,
          admitsRepresentationRoot(gate, RepresentationRootVariant::GateNetlist,
                                   std::nullopt),
          "gate descriptor lost its stageless GateNetlist admission");
  for (RepresentationRootVariant variant :
       {RepresentationRootVariant::AsicPhysical,
        RepresentationRootVariant::FpgaPhysical,
        RepresentationRootVariant::FpgaImage}) {
    require(__func__, !admitsRepresentationRoot(rtl, variant, std::nullopt),
            "RTL descriptor admitted a physical root");
    require(__func__, !admitsRepresentationRoot(gate, variant, std::nullopt),
            "gate descriptor admitted a physical root");
  }

  const ImplementationRepresentationRoot rtlRoot =
      makeRoot(RepresentationRootVariant::Rtl, std::nullopt,
               rtlFormat(__func__), RepresentationObjectKind::Module, __func__);
  require(__func__, !admissionFails(__func__, rtl, rtlRoot),
          "RTL root was rejected by its own descriptor");
  require(__func__, admissionFails(__func__, gate, rtlRoot),
          "RTL root was admitted by the gate descriptor");

  const ImplementationRepresentationRoot physicalRoot =
      makeRoot(RepresentationRootVariant::AsicPhysical,
               RepresentationPhysicalStage::Placed, rtlFormat(__func__),
               RepresentationObjectKind::PhysicalObject, __func__);
  require(__func__, admissionFails(__func__, rtl, physicalRoot),
          "a physical root was admitted without a physical descriptor");

  struct PhysicalCase final {
    RepresentationRootVariant variant;
    std::optional<RepresentationPhysicalStage> stage;
    RepresentationObjectKind topKind;
    std::vector<ImplementationPayload> payloads;
  };
  const std::vector<PhysicalCase> cases{
      {RepresentationRootVariant::AsicPhysical,
       RepresentationPhysicalStage::Placed,
       RepresentationObjectKind::PhysicalObject,
       physicalPayloads(
           {PayloadRole::PhysicalDatabase, PayloadRole::RepresentationIndex})},
      {RepresentationRootVariant::AsicPhysical,
       RepresentationPhysicalStage::Routed,
       RepresentationObjectKind::PhysicalObject,
       physicalPayloads({PayloadRole::PhysicalDatabase,
                         PayloadRole::LayoutStream,
                         PayloadRole::RepresentationIndex})},
      {RepresentationRootVariant::AsicPhysical,
       RepresentationPhysicalStage::Extracted,
       RepresentationObjectKind::PhysicalObject,
       physicalPayloads({PayloadRole::PhysicalDatabase, PayloadRole::Parasitics,
                         PayloadRole::RepresentationIndex})},
      {RepresentationRootVariant::FpgaPhysical,
       RepresentationPhysicalStage::Placed,
       RepresentationObjectKind::DeviceResource,
       physicalPayloads(
           {PayloadRole::PhysicalDatabase, PayloadRole::RepresentationIndex})},
      {RepresentationRootVariant::FpgaPhysical,
       RepresentationPhysicalStage::Routed,
       RepresentationObjectKind::DeviceResource,
       physicalPayloads(
           {PayloadRole::PhysicalDatabase, PayloadRole::RepresentationIndex})},
      {RepresentationRootVariant::FpgaImage, std::nullopt,
       RepresentationObjectKind::DeviceResource,
       physicalPayloads(
           {PayloadRole::DeviceImage, PayloadRole::RepresentationIndex})},
  };
  for (const PhysicalCase &entry : cases) {
    const ImplementationRepresentationRoot root =
        take(__func__, createImplementationRepresentationRoot(
                           entry.variant, entry.stage, physicalFormat(__func__),
                           {entry.topKind, "top"}, entry.payloads));
    require(__func__, !admissionFails(__func__, physical, root),
            "indexed-physical descriptor rejected an exact physical root");
  }

  auto rejectPhysicalRoles =
      [&](RepresentationRootVariant variant,
          std::optional<RepresentationPhysicalStage> stage,
          RepresentationObjectKind topKind,
          std::initializer_list<PayloadRole> roles) {
        const ImplementationRepresentationRoot root =
            take(__func__, createImplementationRepresentationRoot(
                               variant, stage, physicalFormat(__func__),
                               {topKind, "top"}, physicalPayloads(roles)));
        require(
            __func__, admissionFails(__func__, physical, root),
            "indexed-physical descriptor admitted the wrong payload closure");
      };
  rejectPhysicalRoles(RepresentationRootVariant::AsicPhysical,
                      RepresentationPhysicalStage::Placed,
                      RepresentationObjectKind::PhysicalObject,
                      {PayloadRole::PhysicalDatabase});
  rejectPhysicalRoles(RepresentationRootVariant::AsicPhysical,
                      RepresentationPhysicalStage::Placed,
                      RepresentationObjectKind::PhysicalObject,
                      {PayloadRole::PhysicalDatabase, PayloadRole::Parasitics,
                       PayloadRole::RepresentationIndex});
  rejectPhysicalRoles(
      RepresentationRootVariant::AsicPhysical,
      RepresentationPhysicalStage::Extracted,
      RepresentationObjectKind::PhysicalObject,
      {PayloadRole::PhysicalDatabase, PayloadRole::RepresentationIndex});
  rejectPhysicalRoles(RepresentationRootVariant::FpgaPhysical,
                      RepresentationPhysicalStage::Routed,
                      RepresentationObjectKind::DeviceResource,
                      {PayloadRole::PhysicalDatabase, PayloadRole::LayoutStream,
                       PayloadRole::RepresentationIndex});
  rejectPhysicalRoles(RepresentationRootVariant::FpgaImage, std::nullopt,
                      RepresentationObjectKind::DeviceResource,
                      {PayloadRole::DeviceImage, PayloadRole::PhysicalDatabase,
                       PayloadRole::RepresentationIndex});
}

void jsonIsStrictlyCanonical() {
  const std::string json =
      take(__func__, serializeImplementationRepresentationRootJson(
                         makeRoot(RepresentationRootVariant::Rtl, std::nullopt,
                                  rtlFormat(__func__),
                                  RepresentationObjectKind::Module, __func__)));
  auto expectRejection = [&](llvm::StringRef text, llvm::StringRef expected) {
    expectError(__func__, parseImplementationRepresentationRootJson(text),
                expected);
  };
  expectRejection("{\"variant\":\"Rtl\"}", "format_ref");
  expectRejection(
      "{\"variant\":\"rtl\",\"format_ref\":{\"registry\":\"loom.hardware_"
      "representation_format\",\"major\":2,\"minor\":2,\"kind\":0},\"top\":{"
      "\"object_kind\":\"Module\",\"canonical_name\":\"top\"},\"payloads\":[]}",
      "variant");
  expectRejection(
      "{\"variant\":\"Rtl\",\"stage\":null,\"format_ref\":{\"registry\":\"loom."
      "hardware_representation_format\",\"major\":2,\"minor\":2,\"kind\":0}}",
      "stage");
  expectRejection(json + " ", "canonical");

  const std::string physicalJson = take(
      __func__, serializeImplementationRepresentationRootJson(makeRoot(
                    RepresentationRootVariant::FpgaPhysical,
                    RepresentationPhysicalStage::Placed, rtlFormat(__func__),
                    RepresentationObjectKind::DeviceResource, __func__)));
  expectRejection(
      "{\"format_ref\":{\"registry\":\"loom.hardware_representation_format\","
      "\"major\":2,\"minor\":2,\"kind\":0},\"variant\":\"FpgaPhysical\","
      "\"stage\":\"Placed\",\"top\":{\"object_kind\":\"DeviceResource\","
      "\"canonical_name\":\"top\"},\"payloads\":[{\"role\":\"RtlSource\","
      "\"canonical_logical_name\":\"rtl/top.sv\",\"blob_digest\":\"" +
          formatBlobDigestHex(digest("rtl")) +
          "\"},{\"role\":"
          "\"BlackBoxContract\",\"canonical_logical_name\":\"ip/pll.bb\","
          "\"blob_digest\":\"" +
          formatBlobDigestHex(digest("ip")) + "\"}]}",
      "canonical");
  require(
      __func__,
      take(__func__, parseImplementationRepresentationRootJson(physicalJson)) ==
          makeRoot(RepresentationRootVariant::FpgaPhysical,
                   RepresentationPhysicalStage::Placed, rtlFormat(__func__),
                   RepresentationObjectKind::DeviceResource, __func__),
      "physical root JSON did not round-trip");
}

} // namespace

int main() {
  closedVariantsRoundTripExactly();
  binaryFramingIsStrict();
  variantStageLegalityIsClosedAtAuthoring();
  topKindMatchesVariant();
  payloadCatalogIsCanonicalAndUnique();
  descriptorAdmissionIsDataDriven();
  jsonIsStrictlyCanonical();
  return EXIT_SUCCESS;
}
