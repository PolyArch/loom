#include "Fabric/IR/PhysicalTag.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <cstdint>
#include <cstdlib>

namespace {

[[noreturn]] void fail(llvm::StringRef message) {
  llvm::errs() << "physical tag test: " << message << '\n';
  std::exit(1);
}

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

template <typename T> bool rejected(llvm::Expected<T> value) {
  if (value)
    return false;
  llvm::consumeError(value.takeError());
  return true;
}

void roundTripsExactOwnerWidth() {
  const llvm::APInt value(9, 0x101);
  const std::array<std::uint8_t, 2> expected = {0x01, 0x01};
  const auto encoded = take(fabric::encodePhysicalTagValue(9, value));
  if (!llvm::equal(encoded, expected))
    fail("fixed-width big-endian encoding changed");
  const llvm::APInt decoded = take(fabric::decodePhysicalTagValue(9, encoded));
  if (decoded.getBitWidth() != 9 || decoded != value)
    fail("exact owner width did not round trip");

  const auto nibble =
      take(fabric::encodePhysicalTagValue(4, llvm::APInt(4, 10)));
  if (!llvm::equal(nibble, std::array<std::uint8_t, 1>{0x0a}))
    fail("sub-byte tag encoding changed");
}

void rejectsNoncanonicalOrUnrepresentableValues() {
  if (!rejected(fabric::encodePhysicalTagValue(4, llvm::APInt(5, 16))))
    fail("encoder accepted an unrepresentable value");
  if (!rejected(
          fabric::decodePhysicalTagValue(9, std::array<std::uint8_t, 1>{0x01})))
    fail("decoder accepted the wrong byte count");
  if (!rejected(fabric::decodePhysicalTagValue(
          9, std::array<std::uint8_t, 2>{0x80, 0x00})))
    fail("decoder accepted nonzero high padding");
  if (!rejected(fabric::encodePhysicalTagValue(0, llvm::APInt(1, 0))) ||
      !rejected(
          fabric::decodePhysicalTagValue(0, llvm::ArrayRef<std::uint8_t>())))
    fail("zero-width Physical Tag was accepted");
}

} // namespace

int main() {
  roundTripsExactOwnerWidth();
  rejectsNoncanonicalOrUnrepresentableValues();
  llvm::outs() << "physical tag tests passed\n";
  return 0;
}
