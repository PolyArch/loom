#include "Hardware/Implementation/ImplementationPayload.h"

#include "Common/BlobDigest.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
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
    fail(test, "accepted an invalid implementation payload");
  const std::string message = llvm::toString(value.takeError());
  require(test, llvm::StringRef(message).contains(expected), message);
}

void expectError(llvm::StringRef test, llvm::Error error,
                 llvm::StringRef expected) {
  if (!error)
    fail(test, "accepted an invalid implementation payload");
  const std::string message = llvm::toString(std::move(error));
  require(test, llvm::StringRef(message).contains(expected), message);
}

BlobDigest digest(llvm::StringRef contents) {
  return computeBlobDigest(
      llvm::ArrayRef(reinterpret_cast<const std::uint8_t *>(contents.data()),
                     contents.size()));
}

void appendU32Be(std::vector<std::uint8_t> &bytes, std::uint32_t value) {
  bytes.push_back(static_cast<std::uint8_t>(value >> 24));
  bytes.push_back(static_cast<std::uint8_t>(value >> 16));
  bytes.push_back(static_cast<std::uint8_t>(value >> 8));
  bytes.push_back(static_cast<std::uint8_t>(value));
}

void appendU64Be(std::vector<std::uint8_t> &bytes, std::uint64_t value) {
  for (unsigned shift = 56; shift != 0; shift -= 8)
    bytes.push_back(static_cast<std::uint8_t>(value >> shift));
  bytes.push_back(static_cast<std::uint8_t>(value));
}

std::vector<std::uint8_t> expectedBytes(std::uint32_t role,
                                        llvm::StringRef logicalName,
                                        const BlobDigest &blobDigest) {
  std::vector<std::uint8_t> bytes;
  appendU32Be(bytes, role);
  appendU64Be(bytes, logicalName.size());
  bytes.insert(bytes.end(), logicalName.bytes_begin(), logicalName.bytes_end());
  bytes.insert(bytes.end(), blobDigest.bytes().begin(),
               blobDigest.bytes().end());
  return bytes;
}

void stableTagsAndBinaryCodecAreClosed() {
  constexpr std::array<PayloadRole, 9> roles{
      PayloadRole::RtlSource,
      PayloadRole::Netlist,
      PayloadRole::PhysicalDatabase,
      PayloadRole::Parasitics,
      PayloadRole::LayoutStream,
      PayloadRole::DeviceImage,
      PayloadRole::GenerationConstraint,
      PayloadRole::BlackBoxContract,
      PayloadRole::RepresentationIndex,
  };
  for (std::uint32_t tag = 0; tag < roles.size(); ++tag) {
    const ImplementationPayload payload{roles[tag], "rtl/top.sv",
                                        digest(llvm::Twine(tag).str())};
    const std::vector<std::uint8_t> bytes =
        take(__func__, encodeImplementationPayload(payload));
    require(__func__,
            bytes == expectedBytes(tag, payload.canonicalLogicalName,
                                   payload.blobDigest),
            "implementation payload binary framing changed");
    require(__func__,
            take(__func__, decodeImplementationPayload(bytes)) == payload,
            "implementation payload binary codec did not round-trip");
  }

  const ImplementationPayload payload{PayloadRole::RtlSource, "rtl/top.sv",
                                      digest("rtl")};
  const std::vector<std::uint8_t> bytes =
      take(__func__, encodeImplementationPayload(payload));
  for (std::size_t size = 0; size < bytes.size(); ++size)
    expectError(
        __func__,
        decodeImplementationPayload(llvm::ArrayRef(bytes).take_front(size)),
        "truncated");

  std::vector<std::uint8_t> trailing = bytes;
  trailing.push_back(0);
  expectError(__func__, decodeImplementationPayload(trailing), "trailing");

  std::vector<std::uint8_t> unknownRole = bytes;
  unknownRole[3] = 9;
  expectError(__func__, decodeImplementationPayload(unknownRole), "role");
  expectError(__func__,
              encodeImplementationPayload(
                  {static_cast<PayloadRole>(9), "rtl/top.sv", digest("rtl")}),
              "role");

  std::vector<std::uint8_t> excessiveLength = bytes;
  std::fill(excessiveLength.begin() + 4, excessiveLength.begin() + 12, 0xff);
  expectError(__func__, decodeImplementationPayload(excessiveLength),
              "truncated");

  const ImplementationPayload longName{PayloadRole::Netlist,
                                       std::string(256, 'a'), digest("gate")};
  const std::vector<std::uint8_t> longBytes =
      take(__func__, encodeImplementationPayload(longName));
  require(__func__,
          longBytes.size() == 300 && longBytes[10] == 1 && longBytes[11] == 0,
          "payload logical-name length is not full-width u64be");
  require(__func__,
          take(__func__, decodeImplementationPayload(longBytes)) == longName,
          "multi-byte payload logical-name length did not round-trip");
}

void exactJsonCodecIsClosed() {
  constexpr std::array<llvm::StringLiteral, 9> spellings{
      "RtlSource",
      "Netlist",
      "PhysicalDatabase",
      "Parasitics",
      "LayoutStream",
      "DeviceImage",
      "GenerationConstraint",
      "BlackBoxContract",
      "RepresentationIndex",
  };
  for (std::uint32_t tag = 0; tag < spellings.size(); ++tag) {
    const ImplementationPayload payload{static_cast<PayloadRole>(tag),
                                        "rtl/top.sv", digest("rtl")};
    const std::string expected =
        (llvm::Twine("{\"role\":\"") + spellings[tag] +
         "\",\"canonical_logical_name\":\"rtl/top.sv\","
         "\"blob_digest\":\"" +
         formatBlobDigestHex(payload.blobDigest) + "\"}")
            .str();
    const std::string json =
        take(__func__, serializeImplementationPayloadJson(payload));
    require(__func__, json == expected,
            "implementation payload canonical JSON changed");
    require(__func__,
            take(__func__, parseImplementationPayloadJson(json)) == payload,
            "implementation payload JSON codec did not round-trip");
  }

  expectError(
      __func__,
      parseImplementationPayloadJson(
          R"json({"canonical_logical_name":"rtl/top.sv","role":"RtlSource","blob_digest":"0000000000000000000000000000000000000000000000000000000000000000"})json"),
      "canonical");
  expectError(
      __func__,
      parseImplementationPayloadJson(
          R"json({"role":"RtlSource","canonical_logical_name":"rtl/top.sv","blob_digest":"0000000000000000000000000000000000000000000000000000000000000000","extra":0})json"),
      "field");
  expectError(
      __func__,
      parseImplementationPayloadJson(
          R"json({"role":"rtl_source","canonical_logical_name":"rtl/top.sv","blob_digest":"0000000000000000000000000000000000000000000000000000000000000000"})json"),
      "role");
  expectError(
      __func__,
      parseImplementationPayloadJson(
          R"json({"role":"\u0052tlSource","canonical_logical_name":"rtl/top.sv","blob_digest":"0000000000000000000000000000000000000000000000000000000000000000"})json"),
      "canonical");
  expectError(
      __func__,
      parseImplementationPayloadJson(
          R"json({"role":0,"canonical_logical_name":"rtl/top.sv","blob_digest":"0000000000000000000000000000000000000000000000000000000000000000"})json"),
      "string");
  expectError(
      __func__,
      parseImplementationPayloadJson(
          R"json({"role":"RtlSource","canonical_logical_name":"rtl/top.sv","blob_digest":"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"})json"),
      "digest");
}

void logicalNameGrammarIsClosed() {
  const BlobDigest content = digest("payload");
  const std::string utf8Name = std::string("rtl/") + "\xc3\xa9" + ".sv";
  for (const std::string &name :
       std::vector<std::string>{"top.sv", "rtl/top.sv", "a/b/c", utf8Name})
    if (llvm::Error error = validateImplementationPayload(
            {PayloadRole::RtlSource, name, content}))
      fail(__func__, llvm::toString(std::move(error)));

  for (const std::string &name :
       std::vector<std::string>{"", "/top.sv", "rtl/", "rtl//top.sv", ".", "..",
                                "rtl/./top.sv", "rtl/../top.sv"})
    expectError(
        __func__,
        validateImplementationPayload({PayloadRole::RtlSource, name, content}),
        "logical name");

  std::string nulName("rtl/\0top.sv", 11);
  expectError(
      __func__,
      validateImplementationPayload({PayloadRole::RtlSource, nulName, content}),
      "logical name");
  const std::string invalidUtf8 = std::string("rtl/") + "\xc3\x28";
  expectError(__func__,
              validateImplementationPayload(
                  {PayloadRole::RtlSource, invalidUtf8, content}),
              "UTF-8");
}

void canonicalCatalogHasOneOrderingAndOwner() {
  const BlobDigest digestA = digest("a");
  const BlobDigest digestB = digest("b");
  std::vector<ImplementationPayload> payloads{
      {PayloadRole::Netlist, "z", digestA},
      {PayloadRole::RtlSource, "aa", digestA},
      {PayloadRole::RtlSource, "z", digestA},
  };
  const std::vector<ImplementationPayload> canonical =
      take(__func__, canonicalizeImplementationPayloadCatalog(payloads));
  const std::vector<ImplementationPayload> expected{
      {PayloadRole::RtlSource, "aa", digestA},
      {PayloadRole::RtlSource, "z", digestA},
      {PayloadRole::Netlist, "z", digestA},
  };
  require(__func__, canonical == expected,
          "payload catalog does not use role, name bytes, and digest bytes");
  std::reverse(payloads.begin(), payloads.end());
  require(__func__,
          take(__func__, canonicalizeImplementationPayloadCatalog(payloads)) ==
              canonical,
          "payload catalog depends on author order");

  require(__func__,
          implementationPayloadCanonicalLess(
              {PayloadRole::RtlSource, "aa", digestA},
              {PayloadRole::RtlSource, "z", digestA}),
          "payload ordering uses framed name length instead of name bytes");
  require(__func__,
          implementationPayloadCanonicalLess(
              {PayloadRole::RtlSource, "same", digestA},
              {PayloadRole::RtlSource, "same", digestB}) ==
              (digestA.bytes() < digestB.bytes()),
          "payload ordering omitted the digest-byte tie-break");
  const std::string highNameA = std::string("rtl/") + "\xc2\x80";
  const std::string highNameB = std::string("rtl/") + "\xc3\x80";
  require(__func__,
          implementationPayloadCanonicalLess(
              {PayloadRole::RtlSource, highNameA, digestA},
              {PayloadRole::RtlSource, highNameB, digestA}) &&
              !implementationPayloadCanonicalLess(
                  {PayloadRole::RtlSource, highNameB, digestA},
                  {PayloadRole::RtlSource, highNameA, digestA}),
          "payload ordering does not compare UTF-8 name bytes as unsigned");

  expectError(__func__, canonicalizeImplementationPayloadCatalog({}),
              "nonempty");

  expectError(__func__,
              canonicalizeImplementationPayloadCatalog(
                  {{PayloadRole::RtlSource, "rtl/top.sv", digestA},
                   {PayloadRole::RtlSource, "rtl/top.sv", digestB}}),
              "duplicate");
  expectError(__func__,
              canonicalizeImplementationPayloadCatalog(
                  {{PayloadRole::RtlSource, "rtl/../top.sv", digestA}}),
              "logical name");
}

} // namespace

int main() {
  stableTagsAndBinaryCodecAreClosed();
  exactJsonCodecIsClosed();
  logicalNameGrammarIsClosed();
  canonicalCatalogHasOneOrderingAndOwner();
  return EXIT_SUCCESS;
}
