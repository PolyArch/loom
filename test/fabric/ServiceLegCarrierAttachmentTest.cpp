#include "Fabric/IR/SystemServiceContract.h"

#include "Fabric/Identity/FabricRefBytes.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdlib>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

using namespace loom::fabric;

namespace {

[[noreturn]] void fail(const llvm::Twine &message) {
  llvm::errs() << "service-leg carrier attachment: " << message << '\n';
  std::exit(EXIT_FAILURE);
}

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

template <typename T>
void expectRejected(llvm::Expected<T> value, llvm::StringRef diagnostic) {
  if (value)
    fail("accepted invalid attachment record");
  const std::string message = llvm::toString(value.takeError());
  if (!llvm::StringRef(message).contains(diagnostic))
    fail("unexpected diagnostic: " + message);
}

FabricMemoryEndpointRef memoryEndpoint(FabricEntityId owner) {
  return {FabricMemoryEndpointOwnerRef::of(SystemServiceEndpointRef(owner)), 0};
}

FabricTransportEndpointRef carrier(FabricEntityId owner,
                                   FabricOrdinal ordinal) {
  return {FabricTransportEndpointOwnerRef::of(
              SystemTransportResourceRef(owner)),
          ordinal};
}

std::vector<std::uint8_t> swapReferencePayloads(
    std::vector<std::uint8_t> bytes,
    const FabricTransportEndpointRef &first,
    const FabricTransportEndpointRef &second) {
  const std::vector<std::uint8_t> firstBytes = canonicalFabricBytes(first);
  const std::vector<std::uint8_t> secondBytes = canonicalFabricBytes(second);
  if (firstBytes.size() != secondBytes.size())
    fail("test carrier references have different canonical sizes");
  auto firstAt = std::search(bytes.begin(), bytes.end(), firstBytes.begin(),
                             firstBytes.end());
  auto secondAt = std::search(bytes.begin(), bytes.end(), secondBytes.begin(),
                              secondBytes.end());
  if (firstAt == bytes.end() || secondAt == bytes.end() || firstAt == secondAt)
    fail("could not locate both carrier reference payloads");
  std::swap_ranges(firstAt, firstAt + firstBytes.size(), secondAt);
  return bytes;
}

template <typename T, typename = void>
struct HasEndpointRole : std::false_type {};
template <typename T>
struct HasEndpointRole<
    T, std::void_t<decltype(std::declval<const T &>().role())>>
    : std::true_type {};

template <typename T, typename = void>
struct HasPayloadWidth : std::false_type {};
template <typename T>
struct HasPayloadWidth<
    T, std::void_t<decltype(std::declval<const T &>().payloadWidthBits())>>
    : std::true_type {};

template <typename T, typename = void>
struct HasProtocolName : std::false_type {};
template <typename T>
struct HasProtocolName<
    T, std::void_t<decltype(std::declval<const T &>().protocolName())>>
    : std::true_type {};

template <typename T, typename = void>
struct HasCapabilityOrdinal : std::false_type {};
template <typename T>
struct HasCapabilityOrdinal<
    T, std::void_t<decltype(std::declval<const T &>().capabilityOrdinal())>>
    : std::true_type {};

static_assert(!HasEndpointRole<ServiceLegCarrierAttachmentRecord>::value);
static_assert(!HasPayloadWidth<ServiceLegCarrierAttachmentRecord>::value);
static_assert(!HasProtocolName<ServiceLegCarrierAttachmentRecord>::value);
static_assert(!HasCapabilityOrdinal<ServiceLegCarrierAttachmentRecord>::value);

void checkCanonicalRecord() {
  using Kind = dataflow::semantics::ServiceKind;
  const FabricMemoryEndpointRef endpoint = memoryEndpoint(7);
  const FabricTransportEndpointRef first = carrier(9, 0);
  const FabricTransportEndpointRef second = carrier(9, 1);

  ServiceLegCarrierAttachmentRecord record = take(
      ServiceLegCarrierAttachmentRecord::create(
          endpoint, Kind::MemoryRead, 1, {second, first, first}));
  if (record.endpoint() != endpoint || record.kind() != Kind::MemoryRead ||
      record.legOrdinal() != 1 || record.carriers().size() != 2 ||
      canonicalFabricBytes(record.carriers()[0]) >=
          canonicalFabricBytes(record.carriers()[1]))
    fail("authoring did not normalize the carrier set");

  const std::vector<std::uint8_t> bytes =
      take(encodeServiceLegCarrierAttachmentRecord(record));
  ServiceLegCarrierAttachmentRecord decoded =
      take(decodeServiceLegCarrierAttachmentRecord(bytes));
  if (take(encodeServiceLegCarrierAttachmentRecord(decoded)) != bytes)
    fail("canonical roundtrip changed attachment bytes");

  expectRejected(decodeServiceLegCarrierAttachmentRecord(
                     swapReferencePayloads(bytes, first, second)),
                 "sorted and unique");

  expectRejected(ServiceLegCarrierAttachmentRecord::fromCanonical(
                     endpoint, Kind::MemoryRead, 1, {second, first}),
                 "sorted and unique");
  expectRejected(ServiceLegCarrierAttachmentRecord::fromCanonical(
                     endpoint, Kind::MemoryRead, 1, {first, first}),
                 "sorted and unique");
  expectRejected(ServiceLegCarrierAttachmentRecord::create(
                     endpoint, Kind::MemoryRead, 1, {}),
                 "must not be empty");
  expectRejected(ServiceLegCarrierAttachmentRecord::create(
                     endpoint, Kind::MessageTransfer, 0, {first}),
                 "MessageTransfer");

  constexpr dataflow::StructuralOrdinal wideOrdinal =
      dataflow::StructuralOrdinal{1} << 40;
  ServiceLegCarrierAttachmentRecord wide = take(
      ServiceLegCarrierAttachmentRecord::create(
          endpoint, Kind::MemoryWrite, wideOrdinal, {first}));
  ServiceLegCarrierAttachmentRecord wideDecoded = take(
      decodeServiceLegCarrierAttachmentRecord(
          take(encodeServiceLegCarrierAttachmentRecord(wide))));
  if (wideDecoded.legOrdinal() != wideOrdinal)
    fail("codec narrowed the Dataflow structural ordinal");

  ServiceLegCarrierAttachmentRecord reused =
      take(ServiceLegCarrierAttachmentRecord::create(
          endpoint, Kind::MemoryWrite, 0, {first}));
  if (reused.carriers().front() != first)
    fail("one carrier could not serve another service leg");
}

} // namespace

int main() {
  checkCanonicalRecord();
  return EXIT_SUCCESS;
}
