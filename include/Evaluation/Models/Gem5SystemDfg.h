#ifndef LOOM_EVALUATION_MODELS_GEM5SYSTEMDFG_H
#define LOOM_EVALUATION_MODELS_GEM5SYSTEMDFG_H

#include "Evaluation/ModelDescriptor.h"

namespace loom::evaluation::models {

llvm::Error registerGem5SystemDfgProvider();
EvaluationModelDescriptorRef gem5SystemDfgModelDescriptorRef();

} // namespace loom::evaluation::models

#endif // LOOM_EVALUATION_MODELS_GEM5SYSTEMDFG_H
