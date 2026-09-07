#include "Runtime/SpatialInvocationWire.h"

#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

using namespace loom::runtime;

namespace {

[[noreturn]] void fail(const std::string &message) {
  llvm::errs() << "Spatial invocation wire test: " << message << '\n';
  std::exit(EXIT_FAILURE);
}

void require(bool condition, const std::string &message) {
  if (!condition)
    fail(message);
}

void writeLittleEndian(std::vector<std::uint8_t> &bytes, std::size_t offset,
                       std::uint64_t value, std::size_t byteCount) {
  for (std::size_t byte = 0; byte != byteCount; ++byte)
    bytes[offset + byte] = static_cast<std::uint8_t>(value >> (byte * 8));
}

std::uint64_t readLittleEndian(const std::vector<std::uint8_t> &bytes,
                               std::size_t offset, std::size_t byteCount) {
  std::uint64_t value = 0;
  for (std::size_t byte = 0; byte != byteCount; ++byte)
    value |= static_cast<std::uint64_t>(bytes[offset + byte]) << (byte * 8);
  return value;
}

constexpr std::uint64_t kFirstObjectBytes = 24;
constexpr std::uint64_t kSecondObjectBytes = 8;
constexpr std::uint64_t kFirstRootEntity = 5;
constexpr std::uint64_t kSecondRootEntity = 9;
constexpr std::uint64_t kFirstRootByteOffset = 8;
constexpr std::array<std::uint8_t, 4> kFixedValueBits{0x11, 0x22, 0x33, 0x44};

SpatialInvocationWireLayout projectTwoPointLayout() {
  const std::array<std::uint8_t, 32> identity{0xa5};
  const std::vector<std::vector<std::uint64_t>> points{{3}, {7}};
  std::vector<SpatialInvocationValueLayout> values(4);
  values[0].bitCount = 64;
  values[0].pointerTarget =
      SpatialInvocationPointerTarget{0, kFirstRootByteOffset};
  values[1].bitCount = 32;
  values[1].source = SpatialInvocationValueSource::FixedBits;
  values[1].fixedLittleEndianBits.assign(kFixedValueBits.begin(),
                                         kFixedValueBits.end());
  values[2].bitCount = 32;
  values[2].source = SpatialInvocationValueSource::DenseCoordinate;
  values[3].bitCount = 64;
  values[3].pointerTarget = SpatialInvocationPointerTarget{1, 0};
  // The first object rebinds its base per call site, so its byte offsets are
  // written at run time; the second object keeps its captured offsets.
  const std::vector<SpatialInvocationMemoryObjectLayout> objects{
      {kFirstObjectBytes, true}, {kSecondObjectBytes, false}};
  const std::vector<SpatialInvocationMemoryRootBinding> rootBindings{
      {kFirstRootEntity, 0, kFirstRootByteOffset}, {kSecondRootEntity, 1, 0}};
  SpatialInvocationWireLayout layout;
  std::string diagnostic;
  if (!projectSpatialInvocationWireLayout(identity, 11, 13, points, values,
                                          objects, rootBindings, {64}, layout,
                                          diagnostic))
    fail("could not project the invocation wire layout: " + diagnostic);
  return layout;
}

/// The guest writes only the fields the layout reports as dynamic; every other
/// field must already be exact in the per-point template.
void bakedFieldsAreExact() {
  const SpatialInvocationWireLayout layout = projectTwoPointLayout();
  require(layout.pointTemplates.size() == 2, "layout lost a dense point");
  require(layout.pointTemplates[0].size() == layout.pointTemplates[1].size(),
          "dense points disagree on their wire extent");
  require(!layout.valuePayloadOffsets[1] && !layout.valuePayloadOffsets[2],
          "fixed and coordinate payloads are not baked");
  require(layout.valuePayloadOffsets[0] && layout.valuePayloadOffsets[3],
          "runtime value payloads have no wire slot");
  require(layout.valuePointerTargetOffsetOffsets[0] &&
              !layout.valuePointerTargetOffsetOffsets[1] &&
              !layout.valuePointerTargetOffsetOffsets[2] &&
              !layout.valuePointerTargetOffsetOffsets[3],
          "pointer-target offsets are not classified by their object base");
  require(layout.memoryRootByteOffsetOffsets[0] &&
              !layout.memoryRootByteOffsetOffsets[1],
          "memory-root offsets are not classified by their object base");

  std::vector<std::size_t> offsets{layout.memoryAddressOffsets[0],
                                   layout.memoryAddressOffsets[1],
                                   layout.resultAddressOffsets[0],
                                   *layout.valuePayloadOffsets[0],
                                   *layout.valuePayloadOffsets[3],
                                   *layout.valuePointerTargetOffsetOffsets[0],
                                   *layout.memoryRootByteOffsetOffsets[0]};
  for (std::size_t offset : offsets)
    require(offset % spatialInvocationWireAlignment == 0,
            "an invocation wire field is not naturally aligned");

  for (std::size_t point = 0; point != layout.pointTemplates.size(); ++point) {
    SpatialInvocationWire wire;
    std::string diagnostic;
    require(decodeSpatialInvocationWire(layout.pointTemplates[point], wire,
                                        diagnostic),
            "template does not decode: " + diagnostic);
    const std::uint64_t coordinate = point == 0 ? 3 : 7;
    require(wire.denseCoordinates == std::vector<std::uint64_t>{coordinate},
            "template lost its dense coordinate");
    require(std::equal(kFixedValueBits.begin(), kFixedValueBits.end(),
                       wire.values[1].littleEndianBits.begin()),
            "template lost its fixed value payload");
    require(readLittleEndian(wire.values[2].littleEndianBits, 0, 4) ==
                coordinate,
            "template did not bake its dense-coordinate value payload");
    require(wire.memoryRootBindings[1].byteOffset == 0 &&
                wire.memoryRootBindings[0].byteOffset == kFirstRootByteOffset,
            "template did not bake its captured memory-root offsets");
    require(wire.memoryObjects[0].byteCount == kFirstObjectBytes &&
                wire.memoryObjects[1].byteCount == kSecondObjectBytes,
            "template lost a memory object extent");
  }
}

/// The host writes each dynamic field at the layout's byte offset and the
/// engine's decoder must read exactly that field back.
void dynamicFieldsRoundTripThroughTheirOffsets() {
  const SpatialInvocationWireLayout layout = projectTwoPointLayout();
  std::vector<std::uint8_t> bytes = layout.pointTemplates[1];
  constexpr std::uint64_t kFirstAddress = 0x8000;
  constexpr std::uint64_t kSecondAddress = 0x9000;
  constexpr std::uint64_t kResultAddress = 0xa000;
  constexpr std::uint64_t kRuntimeByteOffset = 16;
  constexpr std::uint64_t kFirstPointerBits = 0x8010;
  constexpr std::uint64_t kSecondPointerBits = 0x9000;
  writeLittleEndian(bytes, layout.memoryAddressOffsets[0], kFirstAddress, 8);
  writeLittleEndian(bytes, layout.memoryAddressOffsets[1], kSecondAddress, 8);
  writeLittleEndian(bytes, layout.resultAddressOffsets[0], kResultAddress, 8);
  writeLittleEndian(bytes, *layout.valuePayloadOffsets[0], kFirstPointerBits,
                    8);
  writeLittleEndian(bytes, *layout.valuePayloadOffsets[3], kSecondPointerBits,
                    8);
  writeLittleEndian(bytes, *layout.valuePointerTargetOffsetOffsets[0],
                    kRuntimeByteOffset, 8);
  writeLittleEndian(bytes, *layout.memoryRootByteOffsetOffsets[0],
                    kRuntimeByteOffset, 8);

  SpatialInvocationWire wire;
  std::string diagnostic;
  require(decodeSpatialInvocationWire(bytes, wire, diagnostic),
          "patched wire does not decode: " + diagnostic);
  require(wire.memoryObjects[0].address == kFirstAddress &&
              wire.memoryObjects[1].address == kSecondAddress,
          "memory address slot does not name its object");
  require(wire.results[0].address == kResultAddress,
          "result address slot does not name its destination");
  require(readLittleEndian(wire.values[0].littleEndianBits, 0, 8) ==
                  kFirstPointerBits &&
              readLittleEndian(wire.values[3].littleEndianBits, 0, 8) ==
                  kSecondPointerBits,
          "runtime value payload slot does not name its value");
  require(wire.values[0].pointerTarget->byteOffset == kRuntimeByteOffset,
          "pointer-target offset slot does not name its byte offset");
  require(wire.memoryRootBindings[0].byteOffset == kRuntimeByteOffset,
          "memory-root offset slot does not name its byte offset");
  require(wire.memoryRootBindings[1].byteOffset == 0,
          "a baked memory-root offset was disturbed");
  require(readLittleEndian(wire.values[2].littleEndianBits, 0, 4) == 7,
          "a baked dense coordinate was disturbed");
}

/// The descriptor owns each object's extent; the Bridge-materialized snapshot
/// must cover exactly that object table.
void memorySnapshotMatchesTheObjectTable() {
  const SpatialInvocationWireLayout layout = projectTwoPointLayout();
  SpatialInvocationWire wire;
  std::string diagnostic;
  require(decodeSpatialInvocationWire(layout.pointTemplates[0], wire,
                                      diagnostic),
          diagnostic);
  std::size_t byteCount = 0;
  require(spatialInvocationMemorySnapshotByteCount(wire, byteCount) &&
              byteCount == kFirstObjectBytes + kSecondObjectBytes,
          "snapshot extent differs from the object table");
  require(spatialInvocationMemorySnapshotOffset(wire, 0) == 0 &&
              spatialInvocationMemorySnapshotOffset(wire, 1) ==
                  kFirstObjectBytes,
          "snapshot object offsets are not the object table prefix sums");
  std::vector<std::uint8_t> snapshot(byteCount, 0x5a);
  require(validateSpatialInvocationMemorySnapshot(wire, snapshot, diagnostic),
          "exact snapshot was rejected: " + diagnostic);
  snapshot.pop_back();
  diagnostic.clear();
  require(!validateSpatialInvocationMemorySnapshot(wire, snapshot,
                                                   diagnostic) &&
              diagnostic.find("does not cover") != std::string::npos,
          "a truncated snapshot was accepted");
}

/// The Bridge byte-compares the invocation and the snapshot it supplied
/// against the plane the engine echoes back.
void resultWireRetainsBothImmutablePlanes() {
  const SpatialInvocationWireLayout layout = projectTwoPointLayout();
  SpatialInvocationResultWire result;
  result.sessionEntryOrdinal = 3;
  result.invocation = layout.pointTemplates[0];
  result.memorySnapshot.assign(kFirstObjectBytes + kSecondObjectBytes, 0x27);
  SpatialInvocationRuntimeInputSnapshot runtimeInput;
  runtimeInput.identity.fill(0x31);
  runtimeInput.canonicalBytes = {1, 2, 3, 4, 5};
  result.runtimeInput = runtimeInput;
  result.spatialBoundaryResult = {9, 8, 7};

  const std::vector<std::uint8_t> encoded =
      encodeSpatialInvocationResultWire(result);
  require(!encoded.empty(), "could not encode the invocation result");
  SpatialInvocationResultWire decoded;
  std::string diagnostic;
  require(decodeSpatialInvocationResultWire(encoded, decoded, diagnostic),
          "invocation result does not decode: " + diagnostic);
  require(decoded.sessionEntryOrdinal == result.sessionEntryOrdinal &&
              decoded.invocation == result.invocation &&
              decoded.memorySnapshot == result.memorySnapshot &&
              decoded.runtimeInput &&
              decoded.runtimeInput->identity == runtimeInput.identity &&
              decoded.runtimeInput->canonicalBytes ==
                  runtimeInput.canonicalBytes &&
              decoded.spatialBoundaryResult == result.spatialBoundaryResult,
          "invocation result round-trip changed a semantic plane");

  SpatialInvocationResultWire staticResult;
  staticResult.memorySnapshot = {1};
  staticResult.spatialBoundaryResult = {2};
  diagnostic.clear();
  require(!decodeSpatialInvocationResultWire(
              encodeSpatialInvocationResultWire(staticResult), decoded,
              diagnostic) &&
              diagnostic.find("static invocation result") != std::string::npos,
          "a static result carrying a memory snapshot was accepted");
}

} // namespace

int main() {
  bakedFieldsAreExact();
  dynamicFieldsRoundTripThroughTheirOffsets();
  memorySnapshotMatchesTheObjectTable();
  resultWireRetainsBothImmutablePlanes();
  return 0;
}
