#include "Common/BlobDigest.h"

#include "llvm/ADT/Twine.h"
#include "llvm/Support/ErrorHandling.h"

#include <openssl/evp.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

namespace loom {
namespace {

llvm::Error digestError(const llvm::Twine &detail) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "blob_digest_failure: " + detail);
}

} // namespace

struct BlobDigestBuilder::State final {
  EVP_MD_CTX *context = nullptr;
  bool finished = false;

  ~State() {
    if (context)
      EVP_MD_CTX_free(context);
  }
};

BlobDigestBuilder::BlobDigestBuilder(std::unique_ptr<State> state)
    : state_(std::move(state)) {}

BlobDigestBuilder::BlobDigestBuilder(BlobDigestBuilder &&) noexcept = default;
BlobDigestBuilder &
BlobDigestBuilder::operator=(BlobDigestBuilder &&) noexcept = default;
BlobDigestBuilder::~BlobDigestBuilder() = default;

llvm::Expected<BlobDigestBuilder> BlobDigestBuilder::create() {
  auto state = std::make_unique<State>();
  state->context = EVP_MD_CTX_new();
  if (!state->context ||
      EVP_DigestInit_ex(state->context, EVP_sha256(), nullptr) != 1)
    return digestError("cannot initialize SHA-256");
  return BlobDigestBuilder(std::move(state));
}

llvm::Error BlobDigestBuilder::update(llvm::ArrayRef<std::uint8_t> bytes) {
  if (!state_ || state_->finished)
    return digestError("incremental digest is not active");
  if (EVP_DigestUpdate(state_->context, bytes.data(), bytes.size()) != 1)
    return digestError("cannot update SHA-256");
  return llvm::Error::success();
}

llvm::Expected<BlobDigest> BlobDigestBuilder::finish() {
  if (!state_ || state_->finished)
    return digestError("incremental digest is not active");
  BlobDigest::Storage bytes{};
  unsigned size = 0;
  if (EVP_DigestFinal_ex(state_->context, bytes.data(), &size) != 1 ||
      size != bytes.size())
    return digestError("cannot finalize SHA-256");
  state_->finished = true;
  return BlobDigest::fromBytes(bytes);
}

llvm::Expected<BlobDigest>
BlobDigest::fromBytes(llvm::ArrayRef<std::uint8_t> bytes) {
  if (bytes.size() != byteSize)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "blob digest requires exactly 32 bytes");
  Storage storage;
  std::copy(bytes.begin(), bytes.end(), storage.begin());
  return BlobDigest(storage);
}

BlobDigest computeBlobDigest(llvm::ArrayRef<std::uint8_t> logicalBytes) {
  auto builder = BlobDigestBuilder::create();
  if (!builder)
    llvm::report_fatal_error(llvm::Twine(llvm::toString(builder.takeError())));
  if (llvm::Error error = builder->update(logicalBytes))
    llvm::report_fatal_error(
        llvm::Twine(llvm::toString(std::move(error))));
  auto digest = builder->finish();
  if (!digest)
    llvm::report_fatal_error(llvm::Twine(llvm::toString(digest.takeError())));
  return std::move(*digest);
}

std::string formatBlobDigestHex(const BlobDigest &digest) {
  static constexpr char hex[] = "0123456789abcdef";
  std::string result;
  result.reserve(BlobDigest::byteSize * 2);
  for (std::uint8_t byte : digest.bytes()) {
    result.push_back(hex[byte >> 4]);
    result.push_back(hex[byte & 0x0f]);
  }
  return result;
}

llvm::Expected<BlobDigest> parseBlobDigestHex(llvm::StringRef spelling) {
  if (spelling.size() != BlobDigest::byteSize * 2)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "blob digest must use exactly 64 lowercase hexadecimal characters");

  auto parseNibble = [](char character) -> int {
    if (character >= '0' && character <= '9')
      return character - '0';
    if (character >= 'a' && character <= 'f')
      return character - 'a' + 10;
    return -1;
  };

  std::array<std::uint8_t, BlobDigest::byteSize> bytes;
  for (std::size_t index = 0; index < spelling.size(); index += 2) {
    const int high = parseNibble(spelling[index]);
    const int low = parseNibble(spelling[index + 1]);
    if (high < 0 || low < 0)
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "blob digest must use lowercase hexadecimal");
    bytes[index / 2] = static_cast<std::uint8_t>((high << 4) | low);
  }
  return BlobDigest::fromBytes(bytes);
}

} // namespace loom
