#ifndef LOOM_FRONTEND_RAISING_CANDIDATEHINTS_H
#define LOOM_FRONTEND_RAISING_CANDIDATEHINTS_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <string>
#include <system_error>
#include <utility>

namespace llvm {
class Module;
}

namespace loom::raising {

inline constexpr llvm::StringLiteral functionCandidateAnnotationSchema =
    "loom.candidate.function.2.0";
inline constexpr llvm::StringLiteral loopCandidateAnnotationSchema =
    "loom.candidate.loop.1.0";
inline constexpr llvm::StringLiteral loopCandidateMarkerSchema =
    "loom.candidate.loop.marker.1.0";
inline constexpr llvm::StringLiteral candidateTemporaryRetentionAnnotation =
    "loom.candidate.temporary_retention.1.0";
inline constexpr llvm::StringLiteral
    candidateSourceRequiredTemporaryRetentionAnnotation =
        "loom.candidate.temporary_retention.source_required.1.0";
inline constexpr llvm::StringLiteral functionCandidateMetadataName =
    "loom.candidate.function";
inline constexpr llvm::StringLiteral loopCandidateMetadataName =
    "llvm.loop.loom.candidate";
inline constexpr llvm::StringLiteral loopCandidateManifestMetadataName =
    "loom.candidate.loops";
inline constexpr llvm::StringLiteral candidateTemporaryRetentionMetadataName =
    "loom.candidate.temporary_retention";

enum class CandidateHintErrorKind : std::uint8_t {
  InvalidEncoding,
  InvalidPlacement,
  UnsupportedConstruct,
  ProjectionProofNotEstablished,
};

llvm::StringRef candidateHintErrorKindName(CandidateHintErrorKind kind);

class CandidateHintError final : public llvm::ErrorInfo<CandidateHintError> {
public:
  static char ID;

  CandidateHintError(CandidateHintErrorKind kind, std::string message)
      : kind_(kind), message_(std::move(message)) {}

  CandidateHintErrorKind kind() const { return kind_; }
  void log(llvm::raw_ostream &stream) const override;
  std::error_code convertToErrorCode() const override;

private:
  CandidateHintErrorKind kind_;
  std::string message_;
};

struct SourcePosition final {
  std::uint32_t line = 0;
  std::uint32_t column = 0;

  friend bool operator==(const SourcePosition &lhs, const SourcePosition &rhs) {
    return lhs.line == rhs.line && lhs.column == rhs.column;
  }
};

struct FunctionCandidateAnnotation final {
  std::string sourceFile;
  SourcePosition carrier;
  SourcePosition pragma;
  SourcePosition targetBegin;
  SourcePosition targetEnd;

  friend bool operator==(const FunctionCandidateAnnotation &lhs,
                         const FunctionCandidateAnnotation &rhs) {
    return lhs.sourceFile == rhs.sourceFile && lhs.carrier == rhs.carrier &&
           lhs.pragma == rhs.pragma && lhs.targetBegin == rhs.targetBegin &&
           lhs.targetEnd == rhs.targetEnd;
  }
};

struct LoopCandidateAnnotation final {
  std::uint64_t marker = 0;
  std::string sourceFile;
  SourcePosition carrier;
  SourcePosition pragma;
  SourcePosition targetBegin;
  SourcePosition targetEnd;

  friend bool operator==(const LoopCandidateAnnotation &lhs,
                         const LoopCandidateAnnotation &rhs) {
    return lhs.marker == rhs.marker && lhs.sourceFile == rhs.sourceFile &&
           lhs.carrier == rhs.carrier && lhs.pragma == rhs.pragma &&
           lhs.targetBegin == rhs.targetBegin && lhs.targetEnd == rhs.targetEnd;
  }
};

llvm::Expected<std::string>
encodeFunctionCandidateAnnotation(const FunctionCandidateAnnotation &hint);

llvm::Expected<FunctionCandidateAnnotation>
decodeFunctionCandidateAnnotation(llvm::StringRef annotation);

llvm::Expected<std::string>
encodeLoopCandidateAnnotation(const LoopCandidateAnnotation &hint);

llvm::Expected<LoopCandidateAnnotation>
decodeLoopCandidateAnnotation(llvm::StringRef annotation);

std::string encodeLoopCandidateMarker(std::uint64_t marker);
llvm::Expected<std::uint64_t>
decodeLoopCandidateMarker(llvm::StringRef annotation);

llvm::Error removeCandidateTemporaryRetention(llvm::Module &module);

} // namespace loom::raising

#endif // LOOM_FRONTEND_RAISING_CANDIDATEHINTS_H
