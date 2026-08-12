#ifndef LOOM_EVALUATION_CASETEXT_H
#define LOOM_EVALUATION_CASETEXT_H

#include "Common/ArtifactText.h"
#include "Evaluation/Case.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"

namespace loom::evaluation {

void writeEncodedArtifactLocalReferenceJson(
    llvm::json::OStream &json, const EncodedArtifactLocalReference &reference);
llvm::Expected<EncodedArtifactLocalReference>
parseEncodedArtifactLocalReferenceJson(const llvm::json::Object &object);

void writeSubjectTargetRefJson(llvm::json::OStream &json,
                               const SubjectTargetRef &target);
llvm::Expected<SubjectTargetRef>
parseSubjectTargetRefJson(const llvm::json::Value &value);

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
