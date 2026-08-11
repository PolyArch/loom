#ifndef LOOM_EVALUATION_MODELS_FABRICLOWCONFIDENCE_H
#define LOOM_EVALUATION_MODELS_FABRICLOWCONFIDENCE_H

#include "Evaluation/ModelDescriptor.h"

namespace loom::evaluation::models {

llvm::Error registerFabricLowConfidenceProvider();
EvaluationModelDescriptorRef fabricLowConfidenceModelDescriptorRef();

} // namespace loom::evaluation::models

#endif // LOOM_EVALUATION_MODELS_FABRICLOWCONFIDENCE_H
