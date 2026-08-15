#ifndef LOOM_PNR_PNRGENERATION_H
#define LOOM_PNR_PNRGENERATION_H

#include <cstdint>

namespace loom::pnr {

/// Completeness of one provider's configured finite PnR work, independent of
/// the validity of every candidate already published by that invocation.
enum class PnrGenerationTermination : std::uint8_t {
  FixedAttemptsCompleted,
  SemanticLimitReached,
  ProofNotEstablished,
};

} // namespace loom::pnr

#endif // LOOM_PNR_PNRGENERATION_H
