#ifndef LOOM_RUNTIME_GEM5BRIDGEABI_H
#define LOOM_RUNTIME_GEM5BRIDGEABI_H

#include "Runtime/Gem5BridgeWire.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <vector>

namespace loom::runtime {

std::vector<std::uint8_t>
encodeGem5BridgeMessage(const Gem5BridgeMessage &message);

llvm::Expected<Gem5BridgeMessage>
decodeGem5BridgeMessage(llvm::ArrayRef<std::uint8_t> bytes);

} // namespace loom::runtime

#endif // LOOM_RUNTIME_GEM5BRIDGEABI_H
