#include "Runtime/Gem5BridgeABI.h"

#include "llvm/ADT/StringRef.h"

#include <system_error>

namespace loom::runtime {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "gem5_bridge_message_invalid: " + message);
}

} // namespace

std::vector<std::uint8_t>
encodeGem5BridgeMessage(const Gem5BridgeMessage &message) {
  return encodeGem5BridgeWireMessage(message);
}

llvm::Expected<Gem5BridgeMessage>
decodeGem5BridgeMessage(llvm::ArrayRef<std::uint8_t> bytes) {
  std::vector<std::uint8_t> storage(bytes.begin(), bytes.end());
  Gem5BridgeMessage result;
  std::string diagnostic;
  if (!decodeGem5BridgeWireMessage(storage, result, diagnostic))
    return invalid(diagnostic);
  return result;
}

} // namespace loom::runtime
