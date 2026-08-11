#ifndef LOOM_EVALUATION_MODELS_GEM5SYSTEMRTL_H
#define LOOM_EVALUATION_MODELS_GEM5SYSTEMRTL_H

#include "Evaluation/ModelDescriptor.h"

namespace loom::evaluation::models {

llvm::Error registerGem5SystemRtlProvider();
EvaluationModelDescriptorRef gem5SystemRtlModelDescriptorRef();

} // namespace loom::evaluation::models

#endif // LOOM_EVALUATION_MODELS_GEM5SYSTEMRTL_H
