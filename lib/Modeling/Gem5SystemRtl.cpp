#include "Evaluation/Models/Gem5SystemRtl.h"

#include "Evaluation/ModelProvider.h"
#include "Evaluation/ProductionRegistry.h"
#include "Runtime/Gem5SystemExecution.h"

namespace loom::evaluation::models {

EvaluationModelDescriptorRef gem5SystemRtlModelDescriptorRef() {
  return llvm::cantFail(
      builtinEvaluationModelDescriptorRef(BuiltinEvaluationModel::Gem5SystemRtl));
}

llvm::Error registerGem5SystemRtlProvider() {
  static const EvaluationModelProvider provider{
      gem5SystemRtlModelDescriptorRef(),
      EvaluationModelExternalPrepareImportProvider{
          &runtime::prepareGem5SystemInvocation,
          &runtime::importGem5SystemInvocation}};
  return registerEvaluationModelProvider(provider);
}

} // namespace loom::evaluation::models
