#ifndef LOOM_EVALUATION_MODELS_GEM5SYSTEMCGRA_H
#define LOOM_EVALUATION_MODELS_GEM5SYSTEMCGRA_H

#include "Evaluation/ModelDescriptor.h"

namespace loom::evaluation::models {

llvm::Error registerGem5SystemCgraProvider();
EvaluationModelDescriptorRef gem5SystemCgraModelDescriptorRef();

} // namespace loom::evaluation::models

#endif // LOOM_EVALUATION_MODELS_GEM5SYSTEMCGRA_H
