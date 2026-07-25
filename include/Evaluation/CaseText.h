#ifndef LOOM_EVALUATION_CASETEXT_H
#define LOOM_EVALUATION_CASETEXT_H

#include "Evaluation/Case.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"

namespace loom::evaluation {

/// Canonical scope text with fixed field ordering and closed-variant
/// spellings. Every reference carries its exact schema identity and version;
/// the owner-local payload is emitted as lowercase hexadecimal and is never
/// decoded by the Evaluation serializer itself.
void writeEvaluationScopeJson(llvm::json::OStream &json,
                              const EvaluationScope &scope);

/// Strict canonical decoding against the query kind's own scope forms.
/// Unknown fields, a wrong arity, a malformed reference, and a local payload
/// the owner codec rejects are invalid. Case-relative anchor, closure, and
/// pattern checks run where the exact case is known.
llvm::Expected<EvaluationScope>
parseEvaluationScopeJson(const llvm::json::Object &object,
                         llvm::ArrayRef<ScopeFormDescriptor> forms);

} // namespace loom::evaluation

#endif // LOOM_EVALUATION_CASETEXT_H
