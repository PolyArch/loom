#ifndef LOOM_EVALUATION_CONDITIONTEXT_H
#define LOOM_EVALUATION_CONDITIONTEXT_H

#include "Evaluation/Case.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"

namespace loom::evaluation {

void writeEvaluationConditionJson(llvm::json::OStream &json,
                                  const EvaluationCondition &condition);
llvm::Expected<EvaluationCondition>
parseEvaluationConditionJson(const llvm::json::Value &value);

} // namespace loom::evaluation

#endif // LOOM_EVALUATION_CONDITIONTEXT_H
