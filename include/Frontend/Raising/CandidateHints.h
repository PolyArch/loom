#ifndef LOOM_FRONTEND_RAISING_CANDIDATEHINTS_H
#define LOOM_FRONTEND_RAISING_CANDIDATEHINTS_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <string>

namespace loom::raising {

inline constexpr llvm::StringLiteral functionCandidateAnnotationSchema =
    "loom.candidate.function.1.0";

struct SourcePosition final {
  std::uint32_t line = 0;
  std::uint32_t column = 0;

  friend bool operator==(const SourcePosition &lhs, const SourcePosition &rhs) {
    return lhs.line == rhs.line && lhs.column == rhs.column;
  }
};

struct FunctionCandidateAnnotation final {
  std::string sourceFile;
  SourcePosition pragma;
  SourcePosition targetBegin;
  SourcePosition targetEnd;

  friend bool operator==(const FunctionCandidateAnnotation &lhs,
                         const FunctionCandidateAnnotation &rhs) {
    return lhs.sourceFile == rhs.sourceFile && lhs.pragma == rhs.pragma &&
           lhs.targetBegin == rhs.targetBegin && lhs.targetEnd == rhs.targetEnd;
  }
};

llvm::Expected<std::string>
encodeFunctionCandidateAnnotation(const FunctionCandidateAnnotation &hint);

llvm::Expected<FunctionCandidateAnnotation>
decodeFunctionCandidateAnnotation(llvm::StringRef annotation);

} // namespace loom::raising

#endif // LOOM_FRONTEND_RAISING_CANDIDATEHINTS_H
