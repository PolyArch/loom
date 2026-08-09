#include "Hardware/Implementation/PhysicalRepresentationIndex.h"

#include "Common/BlobStore.h"
#include "Hardware/Implementation/ImplementationRepresentationRoot.h"
#include "Hardware/Implementation/RepresentationIndex.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
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
    fail(test, "accepted invalid physical representation index state");
  const std::string message = llvm::toString(value.takeError());
  require(test, llvm::StringRef(message).contains(expected), message);
}

void expectInvalidIndex(llvm::StringRef test,
                        llvm::Expected<RepresentationIndex> value,
                        llvm::StringRef expected) {
  if (value)
    fail(test, "accepted invalid indexed physical representation");
  bool matched = false;
  llvm::Error remainder = llvm::handleErrors(
      value.takeError(), [&](const RepresentationIndexFailure &failure) {
        matched = failure.kind() == RepresentationIndexFailureKind::Invalid &&
                  failure.reason().contains(expected);
        if (!matched)
          fail(test, "unexpected representation-index failure: " +
                         failure.reason().str());
      });
  if (remainder)
    fail(test, llvm::toString(std::move(remainder)));
  require(test, matched, "did not receive the expected Invalid failure");
}

std::vector<std::uint8_t> bytes(llvm::StringRef text) {
  return {text.bytes_begin(), text.bytes_end()};
}

RepresentationFormatDescriptorRef physicalFormat(llvm::StringRef test) {
  return take(test, RepresentationFormatDescriptorRef::get(
                        RepresentationFormatKind::IndexedPhysical));
}

ImplementationPayload putPayload(llvm::StringRef test, const BlobStore &store,
                                 PayloadRole role, llvm::StringRef name,
                                 llvm::StringRef contents) {
  return {role, name.str(), take(test, store.put(bytes(contents)))};
}

PhysicalRepresentationObject object(RepresentationObjectKind kind,
                                    llvm::StringRef name) {
  return {{kind, name.str()}, std::nullopt};
}

PhysicalRepresentationObject terminal(RepresentationObjectKind kind,
                                      llvm::StringRef name,
                                      RepresentationSignalDirection direction,
                                      std::uint64_t bitWidth) {
  return {{kind, name.str()},
          RepresentationSignalGeometry{direction, bitWidth}};
}

struct PublishedPhysicalRoot final {
  ImplementationRepresentationRoot root;
  PhysicalRepresentationIndexPayload index;
  std::string indexBytes;
};

PublishedPhysicalRoot publishRoot(
    llvm::StringRef test, const BlobStore &store,
    RepresentationRootVariant variant,
    std::optional<RepresentationPhysicalStage> stage, RepresentationLocator top,
    std::vector<ImplementationPayload> payloads,
    std::vector<PhysicalRepresentationObject> objects,
    std::vector<RepresentationLocator> unresolved = {},
    llvm::StringRef indexLogicalName = "index/physical.json",
    std::optional<RepresentationFormatDescriptorRef> format = std::nullopt) {
  const RepresentationFormatDescriptorRef selected =
      format.value_or(physicalFormat(test));
  PhysicalRepresentationIndexPayload index =
      take(test, createPhysicalRepresentationIndexPayload(
                     selected, variant, stage, top, indexLogicalName.str(),
                     payloads, std::move(objects), std::move(unresolved)));
  const std::string indexBytes =
      take(test, serializePhysicalRepresentationIndexPayloadJson(index));
  payloads.push_back({PayloadRole::RepresentationIndex, index.indexLogicalName,
                      take(test, store.put(bytes(indexBytes)))});
  return {take(test, createImplementationRepresentationRoot(
                         variant, stage, selected, std::move(top),
                         std::move(payloads))),
          std::move(index), indexBytes};
}

void requireFacts(llvm::StringRef test, const RepresentationIndex &index,
                  const RepresentationLocator &locator,
                  const RepresentationObjectFacts &expected) {
  const std::optional<RepresentationObjectFacts> actual =
      take(test, index.lookup(locator));
  require(test, actual && *actual == expected,
          "indexed physical object facts changed");
}

void requireInvalidLookup(llvm::StringRef test,
                          const RepresentationIndex &index,
                          const RepresentationLocator &locator) {
  auto result = index.lookup(locator);
  if (result)
    fail(test, "accepted a locator outside the exact physical admission");
  bool matched = false;
  llvm::Error remainder = llvm::handleErrors(
      result.takeError(), [&](const RepresentationIndexFailure &failure) {
        matched = failure.kind() == RepresentationIndexFailureKind::Invalid &&
                  failure.reason().contains("object kind");
      });
  if (remainder)
    fail(test, llvm::toString(std::move(remainder)));
  require(test, matched, "lookup returned the wrong admission failure");
}

void codecIsDeterministicAndCanonical(const std::filesystem::path &root) {
  const BlobStore store((root / "codec-blobs").string());
  std::filesystem::create_directories(root / "codec-blobs");
  std::vector<ImplementationPayload> payloads{
      putPayload(__func__, store, PayloadRole::PhysicalDatabase,
                 "database/state.bin", "synthetic physical state")};
  const RepresentationLocator top{RepresentationObjectKind::PhysicalObject,
                                  "chip"};
  const RepresentationLocator unresolved{RepresentationObjectKind::Module,
                                         "memory_macro"};
  PhysicalRepresentationIndexPayload index = take(
      __func__,
      createPhysicalRepresentationIndexPayload(
          physicalFormat(__func__), RepresentationRootVariant::AsicPhysical,
          RepresentationPhysicalStage::Placed, top, "index/physical.json",
          payloads,
          {object(RepresentationObjectKind::PhysicalObject, "chip"),
           terminal(RepresentationObjectKind::Port, "chip.input",
                    RepresentationSignalDirection::Input, 32),
           object(RepresentationObjectKind::Module, "memory_macro")},
          {unresolved}));

  const std::string expected =
      "{\"format_ref\":" +
      serializeRepresentationFormatDescriptorRefJson(physicalFormat(__func__)) +
      ",\"variant\":\"AsicPhysical\",\"stage\":\"Placed\",\"top\":" +
      take(__func__, serializeRepresentationLocatorJson(top)) +
      ",\"index_logical_name\":\"index/physical.json\",\"payloads\":[" +
      take(__func__, serializeImplementationPayloadJson(payloads.front())) +
      "],\"objects\":[{\"locator\":" +
      take(__func__, serializeRepresentationLocatorJson(unresolved)) +
      "},{\"locator\":" +
      take(__func__, serializeRepresentationLocatorJson(
                         {RepresentationObjectKind::Port, "chip.input"})) +
      ",\"signal_geometry\":{\"direction\":\"Input\",\"bit_width\":32}},"
      "{\"locator\":" +
      take(__func__, serializeRepresentationLocatorJson(top)) +
      "}],\"unresolved_external_definitions\":[" +
      take(__func__, serializeRepresentationLocatorJson(unresolved)) + "]}";
  const std::string encoded =
      take(__func__, serializePhysicalRepresentationIndexPayloadJson(index));
  require(__func__, encoded == expected,
          "physical index canonical JSON changed");
  const PhysicalRepresentationIndexPayload decoded =
      take(__func__, parsePhysicalRepresentationIndexPayloadJson(encoded));
  require(__func__, decoded == index,
          "physical index JSON did not decode exactly");
  require(__func__,
          take(__func__, serializePhysicalRepresentationIndexPayloadJson(
                             decoded)) == encoded,
          "physical index decode/re-encode changed bytes");

  PhysicalRepresentationIndexPayload reordered = index;
  std::reverse(reordered.objects.begin(), reordered.objects.end());
  expectError(__func__,
              serializePhysicalRepresentationIndexPayloadJson(reordered),
              "canonical order");
  PhysicalRepresentationIndexPayload duplicate = index;
  duplicate.objects.push_back(duplicate.objects.front());
  expectError(__func__,
              serializePhysicalRepresentationIndexPayloadJson(duplicate),
              "duplicate");
  expectError(__func__,
              parsePhysicalRepresentationIndexPayloadJson(encoded + " "),
              "canonical");

  std::string unknownField = encoded;
  unknownField.insert(unknownField.size() - 1, ",\"extra\":0");
  expectError(__func__,
              parsePhysicalRepresentationIndexPayloadJson(unknownField),
              "unknown field");

  PhysicalRepresentationIndexPayload recursive = index;
  recursive.payloads.push_back({PayloadRole::RepresentationIndex,
                                "index/other.json",
                                computeBlobDigest(bytes("other index"))});
  expectError(__func__,
              serializePhysicalRepresentationIndexPayloadJson(recursive),
              "RepresentationIndex");

  PhysicalRepresentationIndexPayload invalidDirection = index;
  invalidDirection.objects[1].signalGeometry->direction =
      static_cast<RepresentationSignalDirection>(3);
  expectError(__func__,
              serializePhysicalRepresentationIndexPayloadJson(invalidDirection),
              "direction");

  std::string unknownDirection = encoded;
  const std::size_t directionOffset = unknownDirection.find("\"Input\"");
  require(__func__, directionOffset != std::string::npos,
          "fixture direction is absent");
  unknownDirection.replace(directionOffset, 7, "\"Unknown\"");
  expectError(__func__,
              parsePhysicalRepresentationIndexPayloadJson(unknownDirection),
              "direction");

  expectError(
      __func__,
      createPhysicalRepresentationIndexPayload(
          physicalFormat(__func__), RepresentationRootVariant::AsicPhysical,
          RepresentationPhysicalStage::Placed,
          {RepresentationObjectKind::PhysicalObject, "chip.floorplan"},
          "index/nested-asic-top.json", payloads,
          {object(RepresentationObjectKind::PhysicalObject, "chip.floorplan")},
          {}),
      "top");
  const ImplementationPayload image =
      putPayload(__func__, store, PayloadRole::DeviceImage, "image/design.bin",
                 "synthetic image");
  expectError(
      __func__,
      createPhysicalRepresentationIndexPayload(
          physicalFormat(__func__), RepresentationRootVariant::FpgaImage,
          std::nullopt,
          {RepresentationObjectKind::DeviceResource, "device.region"},
          "index/nested-fpga-top.json", {image},
          {object(RepresentationObjectKind::DeviceResource, "device.region")},
          {}),
      "top");
}

void allPhysicalAdmissionsIndexOpaqueClosures(
    const std::filesystem::path &root) {
  struct Case final {
    llvm::StringRef name;
    RepresentationRootVariant variant;
    std::optional<RepresentationPhysicalStage> stage;
    RepresentationObjectKind topKind;
    std::vector<PayloadRole> roles;
  };
  const std::vector<Case> cases{
      {"asic-placed",
       RepresentationRootVariant::AsicPhysical,
       RepresentationPhysicalStage::Placed,
       RepresentationObjectKind::PhysicalObject,
       {PayloadRole::PhysicalDatabase}},
      {"asic-routed",
       RepresentationRootVariant::AsicPhysical,
       RepresentationPhysicalStage::Routed,
       RepresentationObjectKind::PhysicalObject,
       {PayloadRole::PhysicalDatabase, PayloadRole::LayoutStream}},
      {"asic-extracted",
       RepresentationRootVariant::AsicPhysical,
       RepresentationPhysicalStage::Extracted,
       RepresentationObjectKind::PhysicalObject,
       {PayloadRole::PhysicalDatabase, PayloadRole::Parasitics}},
      {"fpga-placed",
       RepresentationRootVariant::FpgaPhysical,
       RepresentationPhysicalStage::Placed,
       RepresentationObjectKind::DeviceResource,
       {PayloadRole::PhysicalDatabase}},
      {"fpga-routed",
       RepresentationRootVariant::FpgaPhysical,
       RepresentationPhysicalStage::Routed,
       RepresentationObjectKind::DeviceResource,
       {PayloadRole::PhysicalDatabase}},
      {"fpga-image",
       RepresentationRootVariant::FpgaImage,
       std::nullopt,
       RepresentationObjectKind::DeviceResource,
       {PayloadRole::DeviceImage}},
  };

  for (const Case &entry : cases) {
    const std::filesystem::path storePath = root / entry.name.str();
    std::filesystem::create_directories(storePath);
    const BlobStore store(storePath.string());
    std::vector<ImplementationPayload> payloads;
    for (PayloadRole role : entry.roles) {
      const std::vector<std::uint8_t> opaqueBytes{
          0, '\r', 0xff, static_cast<std::uint8_t>(role)};
      payloads.push_back(
          {role, "payload/" + std::to_string(static_cast<std::uint32_t>(role)),
           take(entry.name, store.put(opaqueBytes))});
    }
    const RepresentationLocator top{entry.topKind, "design"};
    const RepresentationLocator port{RepresentationObjectKind::Port,
                                     "design.input"};
    const RepresentationLocator nested{entry.topKind, "design.resource"};
    const RepresentationLocator unresolved{RepresentationObjectKind::Module,
                                           "external_cell"};
    const PublishedPhysicalRoot published = publishRoot(
        entry.name, store, entry.variant, entry.stage, top, payloads,
        {object(entry.topKind, "design"),
         terminal(RepresentationObjectKind::Port, "design.input",
                  RepresentationSignalDirection::Input, 16),
         object(entry.topKind, "design.resource"),
         object(RepresentationObjectKind::Module, "external_cell")},
        {unresolved});
    const RepresentationIndex index =
        take(entry.name, indexRepresentationRoot(published.root, store));
    require(entry.name,
            index.rootVariant() == entry.variant &&
                index.stage() == entry.stage,
            "physical index lost its exact root claim");
    requireFacts(entry.name, index, top, {entry.topKind, std::nullopt});
    requireFacts(entry.name, index, port,
                 {RepresentationObjectKind::Port,
                  RepresentationSignalGeometry{
                      RepresentationSignalDirection::Input, 16}});
    requireFacts(entry.name, index, nested, {entry.topKind, std::nullopt});
    requireFacts(entry.name, index, unresolved,
                 {RepresentationObjectKind::Module, std::nullopt});
    require(entry.name,
            index.unresolvedExternalDefinitions() ==
                llvm::ArrayRef<RepresentationLocator>(unresolved),
            "physical unresolved-definition inventory changed");
    requireInvalidLookup(
        entry.name, index,
        {entry.topKind == RepresentationObjectKind::PhysicalObject
             ? RepresentationObjectKind::DeviceResource
             : RepresentationObjectKind::PhysicalObject,
         "design.foreign"});
  }
}

PublishedPhysicalRoot basicPlacedRoot(llvm::StringRef test,
                                      const BlobStore &store) {
  const ImplementationPayload database =
      putPayload(test, store, PayloadRole::PhysicalDatabase,
                 "database/state.bin", "synthetic state A");
  return publishRoot(test, store, RepresentationRootVariant::AsicPhysical,
                     RepresentationPhysicalStage::Placed,
                     {RepresentationObjectKind::PhysicalObject, "chip"},
                     {database},
                     {object(RepresentationObjectKind::PhysicalObject, "chip"),
                      terminal(RepresentationObjectKind::Port, "chip.input",
                               RepresentationSignalDirection::Input, 32),
                      object(RepresentationObjectKind::Net, "chip.activity")});
}

void missingStaleForeignAndUndeclaredStateIsRejected(
    const std::filesystem::path &root) {
  const std::filesystem::path storePath = root / "rejection-blobs";
  std::filesystem::create_directories(storePath);
  const BlobStore store(storePath.string());
  const PublishedPhysicalRoot valid = basicPlacedRoot(__func__, store);
  take(__func__, indexRepresentationRoot(valid.root, store));

  ImplementationRepresentationRoot missingIndex = valid.root;
  missingIndex.payloads.erase(
      std::remove_if(missingIndex.payloads.begin(), missingIndex.payloads.end(),
                     [](const ImplementationPayload &payload) {
                       return payload.role == PayloadRole::RepresentationIndex;
                     }),
      missingIndex.payloads.end());
  expectInvalidIndex(__func__, indexRepresentationRoot(missingIndex, store),
                     "cardinality");

  ImplementationRepresentationRoot duplicateIndex = valid.root;
  duplicateIndex.payloads.push_back(
      {PayloadRole::RepresentationIndex, "index/duplicate.json",
       duplicateIndex.payloads.back().blobDigest});
  duplicateIndex =
      take(__func__, createImplementationRepresentationRoot(
                         duplicateIndex.variant, duplicateIndex.stage,
                         duplicateIndex.formatRef, duplicateIndex.top,
                         duplicateIndex.payloads));
  expectInvalidIndex(__func__, indexRepresentationRoot(duplicateIndex, store),
                     "cardinality");

  ImplementationRepresentationRoot stale = valid.root;
  for (ImplementationPayload &payload : stale.payloads)
    if (payload.role == PayloadRole::PhysicalDatabase)
      payload.blobDigest =
          take(__func__, store.put(bytes("synthetic state B")));
  expectInvalidIndex(__func__, indexRepresentationRoot(stale, store),
                     "payload catalog");

  ImplementationRepresentationRoot foreignStage = valid.root;
  foreignStage.stage = RepresentationPhysicalStage::Routed;
  expectInvalidIndex(__func__, indexRepresentationRoot(foreignStage, store),
                     "root claim");

  ImplementationRepresentationRoot foreignTop = valid.root;
  foreignTop.top.canonicalName = "other";
  expectInvalidIndex(__func__, indexRepresentationRoot(foreignTop, store),
                     "top");

  ImplementationRepresentationRoot undeclared = valid.root;
  undeclared.payloads.push_back(
      putPayload(__func__, store, PayloadRole::Netlist, "netlist/extra.v",
                 "synthetic undeclared payload"));
  undeclared = take(__func__, createImplementationRepresentationRoot(
                                  undeclared.variant, undeclared.stage,
                                  undeclared.formatRef, undeclared.top,
                                  undeclared.payloads));
  expectInvalidIndex(__func__, indexRepresentationRoot(undeclared, store),
                     "payload catalog");

  const ImplementationPayload database =
      putPayload(__func__, store, PayloadRole::PhysicalDatabase,
                 "database/routed.bin", "synthetic routed state");
  const ImplementationPayload layout =
      putPayload(__func__, store, PayloadRole::LayoutStream,
                 "layout/stream.bin", "synthetic layout stream");
  PhysicalRepresentationIndexPayload partialIndex = take(
      __func__,
      createPhysicalRepresentationIndexPayload(
          physicalFormat(__func__), RepresentationRootVariant::AsicPhysical,
          RepresentationPhysicalStage::Routed,
          {RepresentationObjectKind::PhysicalObject, "chip"},
          "index/partial.json", {database},
          {object(RepresentationObjectKind::PhysicalObject, "chip")}, {}));
  const std::string partialBytes = take(
      __func__, serializePhysicalRepresentationIndexPayloadJson(partialIndex));
  const ImplementationRepresentationRoot partialRoot =
      take(__func__,
           createImplementationRepresentationRoot(
               RepresentationRootVariant::AsicPhysical,
               RepresentationPhysicalStage::Routed, physicalFormat(__func__),
               {RepresentationObjectKind::PhysicalObject, "chip"},
               {database,
                layout,
                {PayloadRole::RepresentationIndex, "index/partial.json",
                 take(__func__, store.put(bytes(partialBytes)))}}));
  expectInvalidIndex(__func__, indexRepresentationRoot(partialRoot, store),
                     "payload catalog");

  const BlobDigest absentDigest = computeBlobDigest(bytes("absent blob"));
  const ImplementationPayload absent{PayloadRole::PhysicalDatabase,
                                     "database/absent.bin", absentDigest};
  const PublishedPhysicalRoot absentRoot = publishRoot(
      __func__, store, RepresentationRootVariant::AsicPhysical,
      RepresentationPhysicalStage::Placed,
      {RepresentationObjectKind::PhysicalObject, "absent"}, {absent},
      {object(RepresentationObjectKind::PhysicalObject, "absent")}, {},
      "index/absent.json");
  expectInvalidIndex(__func__, indexRepresentationRoot(absentRoot.root, store),
                     "blob_store_missing");
}

void tamperedAndNoncanonicalIndexBytesAreRejected(
    const std::filesystem::path &root) {
  const std::filesystem::path tamperedPath = root / "tampered-blobs";
  std::filesystem::create_directories(tamperedPath);
  const BlobStore tamperedStore(tamperedPath.string());
  const PublishedPhysicalRoot tampered =
      basicPlacedRoot(__func__, tamperedStore);
  const auto indexPayload =
      std::find_if(tampered.root.payloads.begin(), tampered.root.payloads.end(),
                   [](const ImplementationPayload &payload) {
                     return payload.role == PayloadRole::RepresentationIndex;
                   });
  require(__func__, indexPayload != tampered.root.payloads.end(),
          "fixture has no representation index payload");
  std::ofstream corrupt(tamperedPath /
                            formatBlobDigestHex(indexPayload->blobDigest),
                        std::ios::binary | std::ios::trunc);
  corrupt << "tampered";
  corrupt.close();
  expectInvalidIndex(__func__,
                     indexRepresentationRoot(tampered.root, tamperedStore),
                     "blob_store_corruption");

  const std::filesystem::path noncanonicalPath = root / "noncanonical-blobs";
  std::filesystem::create_directories(noncanonicalPath);
  const BlobStore noncanonicalStore(noncanonicalPath.string());
  PublishedPhysicalRoot noncanonical =
      basicPlacedRoot(__func__, noncanonicalStore);
  const std::string noncanonicalBytes = noncanonical.indexBytes + " ";
  for (ImplementationPayload &payload : noncanonical.root.payloads)
    if (payload.role == PayloadRole::RepresentationIndex)
      payload.blobDigest =
          take(__func__, noncanonicalStore.put(bytes(noncanonicalBytes)));
  expectInvalidIndex(
      __func__, indexRepresentationRoot(noncanonical.root, noncanonicalStore),
      "canonical");

  std::string foreignBytes = noncanonical.indexBytes;
  const std::string kind = "\"kind\":2";
  const std::size_t kindOffset = foreignBytes.find(kind);
  require(__func__, kindOffset != std::string::npos,
          "fixture format reference is absent");
  foreignBytes.replace(kindOffset, kind.size(), "\"kind\":0");
  PublishedPhysicalRoot foreign = basicPlacedRoot(__func__, noncanonicalStore);
  for (ImplementationPayload &payload : foreign.root.payloads)
    if (payload.role == PayloadRole::RepresentationIndex)
      payload.blobDigest =
          take(__func__, noncanonicalStore.put(bytes(foreignBytes)));
  expectInvalidIndex(__func__,
                     indexRepresentationRoot(foreign.root, noncanonicalStore),
                     "format");
}

void noncanonicalIndexTextIsRejected(const std::filesystem::path &root) {
  const std::filesystem::path storePath = root / "text-policy-blobs";
  std::filesystem::create_directories(storePath);
  const BlobStore store(storePath.string());
  const PublishedPhysicalRoot valid = basicPlacedRoot(__func__, store);

  const auto expectRejected = [&](llvm::StringRef name,
                                  std::vector<std::uint8_t> contents,
                                  llvm::StringRef expected) {
    ImplementationRepresentationRoot rootWithInvalidText = valid.root;
    for (ImplementationPayload &payload : rootWithInvalidText.payloads)
      if (payload.role == PayloadRole::RepresentationIndex)
        payload.blobDigest = take(name, store.put(contents));
    expectInvalidIndex(
        name, indexRepresentationRoot(rootWithInvalidText, store), expected);
  };

  std::vector<std::uint8_t> nul = bytes(valid.indexBytes);
  nul.push_back(0);
  expectRejected("index-nul", std::move(nul), "NUL byte");

  std::vector<std::uint8_t> carriageReturn = bytes(valid.indexBytes);
  carriageReturn.push_back('\r');
  expectRejected("index-carriage-return", std::move(carriageReturn),
                 "LF line endings");

  std::vector<std::uint8_t> invalidUtf8 = bytes(valid.indexBytes);
  invalidUtf8.push_back(0xff);
  expectRejected("index-invalid-utf8", std::move(invalidUtf8), "valid UTF-8");
}

void indexedDefClosureIsSelfContained(const std::filesystem::path &root) {
  const std::filesystem::path storePath = root / "indexed-def-blobs";
  std::filesystem::create_directories(storePath);
  const BlobStore store(storePath.string());
  const RepresentationFormatDescriptorRef format =
      take(__func__, RepresentationFormatDescriptorRef::get(
                         RepresentationFormatKind::IndexedDefPhysical));
  const std::string netlist = "module top(input a, output y);\n"
                              "  fixture_cell u0(.A(a), .Z(y));\n"
                              "endmodule\n";
  const std::string def =
      "VERSION 5.8 ;\n"
      "DESIGN top ;\n"
      "PINS 2 ;\n"
      "- VPWR + NET power_main + USE POWER + LAYER M4 ( 0 0 ) ( 1 1 ) "
      "+ FIXED ( 2 2 ) N ;\n"
      "- VGND + NET ground_main + USE GROUND + LAYER M4 ( 0 0 ) ( 1 1 ) "
      "+ FIXED ( 4 2 ) N ;\n"
      "END PINS\n"
      "SPECIALNETS 2 ;\n"
      "- power_main + USE POWER + ROUTED M4 ( 2 2 ) ( 8 2 ) ;\n"
      "- ground_main + USE GROUND + ROUTED M4 ( 4 2 ) ( 8 4 ) ;\n"
      "END SPECIALNETS\n"
      "NETS 1 ;\n"
      "- signal_a ( u0 A ) ( PIN a ) + ROUTED M2 ( 1 1 ) ( 2 2 ) ;\n"
      "END NETS\n"
      "END DESIGN\n";
  std::vector<ImplementationPayload> payloads{
      putPayload(__func__, store, PayloadRole::Netlist, "netlist/top.v",
                 netlist),
      putPayload(__func__, store, PayloadRole::PhysicalDatabase,
                 "database/top.def", def),
      putPayload(__func__, store, PayloadRole::GenerationConstraint,
                 "constraints/top.sdc", "create_clock -period 1 clk\n")};
  const RepresentationLocator unresolved{RepresentationObjectKind::Module,
                                         "fixture_cell"};
  PublishedPhysicalRoot valid =
      publishRoot(__func__, store, RepresentationRootVariant::AsicPhysical,
                  RepresentationPhysicalStage::Routed,
                  {RepresentationObjectKind::PhysicalObject, "top"}, payloads,
                  {object(RepresentationObjectKind::PhysicalObject, "top"),
                   terminal(RepresentationObjectKind::Port, "top.a",
                            RepresentationSignalDirection::Input, 1),
                   terminal(RepresentationObjectKind::Port, "top.y",
                            RepresentationSignalDirection::Output, 1),
                   object(RepresentationObjectKind::Module, "fixture_cell")},
                  {unresolved}, "index/physical.json", format);
  take(__func__, indexRepresentationRoot(valid.root, store));

  ImplementationRepresentationRoot foreignClosure = valid.root;
  std::vector<ImplementationPayload> foreignPayloads;
  for (const ImplementationPayload &payload : foreignClosure.payloads)
    if (payload.role != PayloadRole::RepresentationIndex)
      foreignPayloads.push_back(payload);
  PublishedPhysicalRoot foreign = publishRoot(
      __func__, store, RepresentationRootVariant::AsicPhysical,
      RepresentationPhysicalStage::Routed,
      {RepresentationObjectKind::PhysicalObject, "top"}, foreignPayloads,
      {object(RepresentationObjectKind::PhysicalObject, "top")}, {},
      "index/foreign.json", format);
  expectInvalidIndex(__func__, indexRepresentationRoot(foreign.root, store),
                     "unresolved definitions disagree");
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2)
    fail("main", "expected one temporary directory argument");
  const std::filesystem::path root(argv[1]);
  std::filesystem::create_directories(root);
  codecIsDeterministicAndCanonical(root);
  allPhysicalAdmissionsIndexOpaqueClosures(root);
  missingStaleForeignAndUndeclaredStateIsRejected(root);
  tamperedAndNoncanonicalIndexBytesAreRejected(root);
  noncanonicalIndexTextIsRejected(root);
  indexedDefClosureIsSelfContained(root);
  return EXIT_SUCCESS;
}
