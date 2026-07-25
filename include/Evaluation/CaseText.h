#ifndef LOOM_EVALUATION_CASETEXT_H
#define LOOM_EVALUATION_CASETEXT_H

#include "Evaluation/Case.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"

namespace loom::evaluation {

/// Canonical scope bytes with fixed field ordering and enum spellings. The
/// local-target family is spelled by its own Artifact schema identity and
/// version, so Evaluation neither invents nor caches a family name.
void writeEvaluationScopeJson(llvm::json::OStream &json,
                              const EvaluationScope &scope);

/// Strict canonical decoding against the query kind's own scope forms. The
/// accepted local targets declared by a form's role are the only way a family
/// is resolved, so decoding needs no global family catalog. Unknown fields, a
/// wrong arity, an unaccepted target kind, and a malformed family payload are
/// rejected.
llvm::Expected<EvaluationScope>
parseEvaluationScopeJson(const llvm::json::Object &object,
                         llvm::ArrayRef<ScopeFormDescriptor> forms);

} // namespace loom::evaluation

#endif // LOOM_EVALUATION_CASETEXT_H
