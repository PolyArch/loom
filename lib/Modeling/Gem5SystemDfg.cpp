#include "Evaluation/Models/Gem5SystemDfg.h"

#include "Evaluation/ModelProvider.h"
#include "Evaluation/ProductionRegistry.h"
#include "Runtime/Gem5SystemExecution.h"

namespace loom::evaluation::models {

EvaluationModelDescriptorRef gem5SystemDfgModelDescriptorRef() {
  return llvm::cantFail(builtinEvaluationModelDescriptorRef(
      BuiltinEvaluationModel::Gem5SystemDfg));
}

llvm::Error registerGem5SystemDfgProvider() {
  static const EvaluationModelProvider provider{
      gem5SystemDfgModelDescriptorRef(),
      EvaluationModelExternalPrepareImportProvider{
          &runtime::prepareGem5SystemInvocation,
          &runtime::importGem5SystemInvocation,
          &runtime::openGem5SystemInvocationContext}};
  return registerEvaluationModelProvider(provider);
}

} // namespace loom::evaluation::models
