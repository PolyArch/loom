#ifndef LOOM_EVALUATION_MODELS_SYSTEMRUNTIMEPREDICTOR_H
#define LOOM_EVALUATION_MODELS_SYSTEMRUNTIMEPREDICTOR_H

#include "Evaluation/ModelDescriptor.h"

namespace loom::evaluation::models {

llvm::Error registerSystemRuntimePredictorProvider();
EvaluationModelDescriptorRef systemRuntimePredictorModelDescriptorRef();

} // namespace loom::evaluation::models

#endif // LOOM_EVALUATION_MODELS_SYSTEMRUNTIMEPREDICTOR_H
