#include "Config/ResolvedConfig.h"
#include "Frontend/Payload/FrontendConfigView.h"
#include "Frontend/Payload/PayloadCarrier.h"
#include "Frontend/Payload/RelocatableAcceleratorPayload.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/Module.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Plugins/PassPlugin.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

namespace {

class EmbedRelocatablePayloadPass
    : public llvm::PassInfoMixin<EmbedRelocatablePayloadPass> {
public:
  llvm::PreservedAnalyses run(llvm::Module &module,
                              llvm::ModuleAnalysisManager &) {
    llvm::Expected<bool> alreadyGenerated =
        loom::hasGeneratedRelocatablePayloadCarrier(module);
    if (!alreadyGenerated) {
      module.getContext().emitError(
          "cannot inspect relocatable accelerator payload carrier: " +
          llvm::toString(alreadyGenerated.takeError()));
      return llvm::PreservedAnalyses::all();
    }
    if (*alreadyGenerated)
      return llvm::PreservedAnalyses::all();

    llvm::SmallVector<char, 0> bitcode;
    llvm::raw_svector_ostream stream(bitcode);
    llvm::WriteBitcodeToFile(module, stream,
                             /*ShouldPreserveUseListOrder=*/false,
                             /*Index=*/nullptr,
                             /*GenerateHash=*/false);
    const llvm::ArrayRef<std::uint8_t> bytes(
        reinterpret_cast<const std::uint8_t *>(bitcode.data()), bitcode.size());
    llvm::Expected<loom::RelocatableAcceleratorPayload> payload =
        loom::RelocatableAcceleratorPayload::create(
            bytes, loom::projectResolvedFrontendConfigView(
                       loom::defaultResolvedConfig()));
    if (!payload) {
      module.getContext().emitError(
          "cannot create relocatable accelerator payload: " +
          llvm::toString(payload.takeError()));
      return llvm::PreservedAnalyses::all();
    }

    const loom::CanonicalSemanticBytes canonical =
        payload->canonicalSemanticBytes();
    loom::embedRelocatablePayloadCarrier(module, canonical.bytes());
    return llvm::PreservedAnalyses::none();
  }
};

llvm::PassPluginLibraryInfo pluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "LoomRelocatablePayload",
          LLVM_VERSION_STRING, [](llvm::PassBuilder &builder) {
            builder.registerOptimizerLastEPCallback(
                [](llvm::ModulePassManager &manager, llvm::OptimizationLevel,
                   llvm::ThinOrFullLTOPhase) {
                  manager.addPass(EmbedRelocatablePayloadPass());
                });
          }};
}

} // namespace

extern "C" LLVM_ATTRIBUTE_WEAK llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return pluginInfo();
}
