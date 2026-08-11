#ifndef LOOM_RUNTIME_GEM5SIMULATIONBINDINGINTERNAL_H
#define LOOM_RUNTIME_GEM5SIMULATIONBINDINGINTERNAL_H

#include "Runtime/Gem5SimulationBinding.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <string>

namespace loom::runtime::detail {

std::string serializeGem5SimulationBinding(
    const Gem5SimulationBinding &binding);

llvm::Expected<Gem5SimulationBindingDraft>
parseGem5SimulationBinding(llvm::StringRef canonicalJson);

} // namespace loom::runtime::detail

#endif // LOOM_RUNTIME_GEM5SIMULATIONBINDINGINTERNAL_H
