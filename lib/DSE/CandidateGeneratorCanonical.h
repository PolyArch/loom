#ifndef LOOM_LIB_DSE_CANDIDATEGENERATORCANONICAL_H
#define LOOM_LIB_DSE_CANDIDATEGENERATORCANONICAL_H

#include "DSE/CandidateGenerator.h"

#include "llvm/ADT/ArrayRef.h"

#include <cstdint>
#include <vector>

namespace loom::dse::detail {

std::vector<std::uint8_t> encodeCanonicalCandidateGeneratorInputBindings(
    llvm::ArrayRef<CandidateGeneratorInputBinding> bindings);

std::vector<std::uint8_t> encodeCanonicalResolvedCandidateGeneratorBinding(
    const ResolvedCandidateGeneratorBinding &binding);

} // namespace loom::dse::detail

#endif // LOOM_LIB_DSE_CANDIDATEGENERATORCANONICAL_H
