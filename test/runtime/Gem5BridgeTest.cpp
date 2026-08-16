#include "Runtime/Gem5BridgeABI.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

using namespace loom::runtime;

namespace {

[[noreturn]] void fail(const std::string &message) {
  llvm::errs() << "Gem5 bridge ABI test: " << message << '\n';
  std::exit(EXIT_FAILURE);
}

void require(bool condition, const std::string &message) {
  if (!condition)
    fail(message);
}

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

void requireRejected(std::vector<std::uint8_t> bytes,
                     llvm::StringRef expected) {
  auto decoded = decodeGem5BridgeMessage(bytes);
  if (decoded)
    fail("accepted malformed message");
  const std::string diagnostic = llvm::toString(decoded.takeError());
  require(llvm::StringRef(diagnostic).contains(expected), diagnostic);
}

void envelopeRoundTrip() {
  const Gem5BridgeMessage original{Gem5BridgeMessageKind::ChannelTransfer,
                                   0x1020304050607080ULL,
                                   {0x00, 0x7f, 0x80, 0xff}};
  const auto bytes = encodeGem5BridgeMessage(original);
  const auto decoded = take(decodeGem5BridgeMessage(bytes));
  require(decoded.kind == original.kind &&
              decoded.sequence == original.sequence &&
              decoded.payload == original.payload,
          "message round-trip changed semantic fields");

  auto badMagic = bytes;
  badMagic.front() ^= 0xff;
  requireRejected(std::move(badMagic), "wrong ABI magic");

  auto unknownKind = bytes;
  unknownKind[4] = 0;
  unknownKind[5] = 0;
  unknownKind[6] = 0;
  unknownKind[7] = 5;
  requireRejected(std::move(unknownKind), "unknown message kind");

  auto wrongLength = bytes;
  wrongLength[23] += 1;
  requireRejected(std::move(wrongLength), "payload length");
}

void payloadRoundTrips() {
  const Gem5BridgeMemoryRequest write{
      Gem5BridgeMemoryOperation::Write, 17, 19, 0x1000, 4,
      {0x01, 0x02, 0x03, 0x04}};
  Gem5BridgeMemoryRequest decodedWrite;
  std::string diagnostic;
  require(decodeGem5BridgeMemoryRequest(encodeGem5BridgeMemoryRequest(write),
                                        decodedWrite, diagnostic),
          diagnostic);
  require(decodedWrite.operation == write.operation &&
              decodedWrite.readyAfterTicks == write.readyAfterTicks &&
              decodedWrite.requestId == write.requestId &&
              decodedWrite.address == write.address &&
              decodedWrite.size == write.size &&
              decodedWrite.data == write.data,
          "memory request round-trip changed semantic fields");

  auto invalidRead = encodeGem5BridgeMemoryRequest(
      {Gem5BridgeMemoryOperation::Read, 0, 1, 0x2000, 1, {0xff}});
  Gem5BridgeMemoryRequest rejectedRequest;
  diagnostic.clear();
  require(!decodeGem5BridgeMemoryRequest(invalidRead, rejectedRequest,
                                         diagnostic) &&
              diagnostic.find("operation and size") != std::string::npos,
          "memory request decoder accepted read payload bytes");

  const Gem5BridgeMemoryResponse response{19, true, {0xaa, 0xbb}};
  Gem5BridgeMemoryResponse decodedResponse;
  diagnostic.clear();
  require(
      decodeGem5BridgeMemoryResponse(encodeGem5BridgeMemoryResponse(response),
                                     decodedResponse, diagnostic),
      diagnostic);
  require(decodedResponse.requestId == response.requestId &&
              decodedResponse.success == response.success &&
              decodedResponse.data == response.data,
          "memory response round-trip changed semantic fields");

  const Gem5BridgeCompletion completion{23, 0, {0x10, 0x20, 0x30}};
  Gem5BridgeCompletion decodedCompletion;
  diagnostic.clear();
  require(decodeGem5BridgeCompletion(encodeGem5BridgeCompletion(completion),
                                     decodedCompletion, diagnostic),
          diagnostic);
  require(decodedCompletion.readyAfterTicks == completion.readyAfterTicks &&
              decodedCompletion.status == completion.status &&
              decodedCompletion.result == completion.result,
          "completion round-trip changed semantic fields");

  const Gem5BridgeResult result{1,
                                0x1020304050607080ULL,
                                0x8877665544332211ULL,
                                {0x00, 0x7f, 0x80, 0xff}};
  const auto encodedResult = encodeGem5BridgeResult(result);
  Gem5BridgeResult decodedResult;
  diagnostic.clear();
  require(decodeGem5BridgeResult(encodedResult, decodedResult, diagnostic),
          diagnostic);
  require(decodedResult.status == result.status &&
              decodedResult.completionTick == result.completionTick &&
              decodedResult.sequence == result.sequence &&
              decodedResult.result == result.result,
          "normalized bridge result round-trip changed semantic fields");

  auto badResultMagic = encodedResult;
  badResultMagic.front() ^= 0xff;
  diagnostic.clear();
  require(!decodeGem5BridgeResult(badResultMagic, decodedResult, diagnostic) &&
              diagnostic.find("wrong bridge result magic") != std::string::npos,
          "normalized bridge result accepted a bad magic");

  auto badResultLength = encodedResult;
  badResultLength[31] += 1;
  diagnostic.clear();
  require(!decodeGem5BridgeResult(badResultLength, decodedResult, diagnostic) &&
              diagnostic.find("size does not match") != std::string::npos,
          "normalized bridge result accepted a bad payload length");

  const Gem5BridgeResultCollection collection{
      {result, Gem5BridgeResult{
                   0, 0x1020304050607090ULL, 0x8877665544332212ULL, {0x42}}}};
  const auto encodedCollection = encodeGem5BridgeResultCollection(collection);
  Gem5BridgeResultCollection decodedCollection;
  diagnostic.clear();
  require(decodeGem5BridgeResultCollection(encodedCollection, decodedCollection,
                                           diagnostic),
          diagnostic);
  require(decodedCollection.results.size() == 2 &&
              decodedCollection.results[0].result == result.result &&
              decodedCollection.results[1].completionTick ==
                  collection.results[1].completionTick &&
              decodedCollection.results[1].sequence ==
                  collection.results[1].sequence,
          "bridge result collection round-trip changed semantic fields");

  auto badCollectionCount = encodedCollection;
  badCollectionCount[11] += 1;
  diagnostic.clear();
  require(!decodeGem5BridgeResultCollection(badCollectionCount,
                                            decodedCollection, diagnostic) &&
              diagnostic.find("count exceeds") != std::string::npos,
          "bridge result collection accepted an impossible count");

  auto trailingCollection = encodedCollection;
  trailingCollection.push_back(0xff);
  diagnostic.clear();
  require(!decodeGem5BridgeResultCollection(trailingCollection,
                                            decodedCollection, diagnostic) &&
              diagnostic.find("trailing bytes") != std::string::npos,
          "bridge result collection accepted trailing bytes");
}

} // namespace

int main() {
  envelopeRoundTrip();
  payloadRoundTrips();
  return 0;
}
