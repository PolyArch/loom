#ifndef LOOM_LIB_EDA_ADAPTERS_OPENSOURCE_OPENROADCONFIGURATION_H
#define LOOM_LIB_EDA_ADAPTERS_OPENSOURCE_OPENROADCONFIGURATION_H

#include "EDA/Adapters/OpenSource/OpenRoad.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/JSON.h"

namespace loom::eda::open_source::detail {

inline bool isPortableIdentifier(llvm::StringRef value) {
  const auto first = [](char character) {
    return (character >= 'A' && character <= 'Z') ||
           (character >= 'a' && character <= 'z') || character == '_';
  };
  return !value.empty() && first(value.front()) &&
         llvm::all_of(value.drop_front(), [&](char character) {
           return first(character) || (character >= '0' && character <= '9');
         });
}

// Shared placement/external-file parser. The enclosing canonical codec checks
// its complete encoding after this owner validates and canonicalizes the value.
llvm::Expected<OpenRoadPlacedConfig>
parseOpenRoadPlacedConfigObject(const llvm::json::Object &object);

} // namespace loom::eda::open_source::detail
#endif
