#include "Evaluation/Models/Gem5SystemCgra.h"

#include "Evaluation/ModelProvider.h"
#include "Evaluation/ProductionRegistry.h"
#include "Runtime/Gem5SystemExecution.h"

namespace loom::evaluation::models {

EvaluationModelDescriptorRef gem5SystemCgraModelDescriptorRef() {
  return llvm::cantFail(builtinEvaluationModelDescriptorRef(
      BuiltinEvaluationModel::Gem5SystemCgra));
}

llvm::Error registerGem5SystemCgraProvider() {
  static const EvaluationModelProvider provider{
      gem5SystemCgraModelDescriptorRef(),
      EvaluationModelExternalPrepareImportProvider{
          &runtime::prepareGem5SystemInvocation,
          &runtime::importGem5SystemInvocation,
          &runtime::openGem5SystemInvocationContext,
          &runtime::importGem5SystemInvocationWithExecution}};
  return registerEvaluationModelProvider(provider);
}

} // namespace loom::evaluation::models
