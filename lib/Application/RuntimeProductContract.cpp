#include "RuntimeProductContract.h"

#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "Dataflow/IR/OperationSchemaCodec.h"
#include "Deployment/Deployment.h"
#include "Deployment/DeploymentReference.h"
#include "Simulator/SimulationArtifacts.h"

#include "llvm/ADT/STLExtras.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace loom::application::detail {
namespace {

llvm::Error mismatch(const llvm::Twine &message) {
  return llvm::make_error<ApplicationRuntimeManifestError>(
      ApplicationRuntimeManifestErrorReason::ProductContractMismatch,
      message.str());
}

std::optional<std::uint64_t>
definedU64(const sim::CanonicalValueSequence &value) {
  if (value.tokenCount != 1 || value.lanes.size() != 1 ||
      value.lanes.front().state != sim::SemanticState::Defined ||
      value.lanes.front().pointerTarget ||
      value.lanes.front().bits.getBitWidth() != 64)
    return std::nullopt;
  return value.lanes.front().bits.getZExtValue();
}

std::optional<std::uint64_t>
definedU64(const sim::StructuredProgramArgumentSource &source) {
  const auto *value = std::get_if<sim::CanonicalValueSequence>(&source);
  return value ? definedU64(*value) : std::nullopt;
}

} // namespace

llvm::Error verifyRuntimeProductContract(
    const ProductOracleContract &contract,
    const sim::ImportedStructuredProgramSimulationInputs &sourceInputs,
    const sim::ImportedSystemSimulationInputs &activationInputs,
    const deployment::FinalizedDeployment &deployment,
    const ArtifactStore &artifacts, const BlobStore &blobs) {
  if (contract.entryAbi != ProductEntryABI::CachedInputsProfileOutputV1 ||
      contract.entrySymbol.empty() ||
      contract.entrySymbol.find('\0') != std::string::npos ||
      contract.measuredSamples == 0 ||
      contract.measuredOutputBytesPerSample == 0 ||
      contract.warmupSamples >
          std::numeric_limits<std::uint64_t>::max() -
              contract.measuredSamples ||
      contract.measuredSamples >
          std::numeric_limits<std::uint64_t>::max() /
              contract.measuredOutputBytesPerSample)
    return mismatch("product oracle contract is malformed");
  const std::uint64_t expectedOutputBytes =
      contract.measuredSamples * contract.measuredOutputBytesPerSample;
  auto expectedOutput = blobs.get(contract.expectedOutput);
  if (!expectedOutput)
    return mismatch("product oracle blob is unavailable: " +
                    llvm::toString(expectedOutput.takeError()));
  if (expectedOutput->size() != expectedOutputBytes)
    return mismatch("product oracle blob has the wrong byte extent");

  const auto *structuredWorkload = sourceInputs.workload.structuredProgram();
  const auto *structuredRuntime = sourceInputs.runtimeInput.structuredProgram();
  if (!structuredWorkload || !structuredRuntime)
    return mismatch("product source invocation is not Structured Program");
  auto sourceView = sourceInputs.structuredProgram.view();
  if (!sourceView)
    return mismatch("product source program view is unavailable: " +
                    llvm::toString(sourceView.takeError()));
  auto sourceEntry = sourceView->resolve(structuredWorkload->entryRef);
  if (!sourceEntry)
    return mismatch("product source entry cannot be resolved: " +
                    llvm::toString(sourceEntry.takeError()));
  auto sourceFunction = llvm::dyn_cast_or_null<mlir::LLVM::LLVMFuncOp>(
      sourceEntry->operation);
  if (!sourceFunction || sourceFunction.getSymName() != contract.entrySymbol)
    return mismatch("product source entry symbol differs from its contract");
  const mlir::LLVM::LLVMFunctionType sourceFunctionType =
      sourceFunction.getFunctionType();
  const auto sourceReturnType =
      mlir::dyn_cast<mlir::IntegerType>(sourceFunctionType.getReturnType());
  if (!sourceReturnType || sourceReturnType.getWidth() != 32 ||
      !structuredWorkload->observableContract.returnValue)
    return mismatch("product source entry does not expose its i32 status");

  std::size_t pointerCount = 0;
  for (const sim::StructuredProgramArgumentSource &argument :
       structuredWorkload->argumentPlan)
    pointerCount += static_cast<std::size_t>(
        std::holds_alternative<sim::StructuredRuntimeMemoryInput>(argument));
  if (pointerCount == 0 ||
      contract.outputInterfaceOrdinal != pointerCount - 1)
    return mismatch("product output interface is not the final pointer input");
  const std::size_t cachedInputCount = pointerCount - 1;
  if (cachedInputCount >
          (std::numeric_limits<std::size_t>::max() - 4) / 2 ||
      structuredWorkload->argumentPlan.size() != cachedInputCount * 2 + 4 ||
      sourceFunctionType.getNumParams() !=
          structuredWorkload->argumentPlan.size())
    return mismatch("product source argument count differs from its ABI");
  for (const auto indexed : llvm::enumerate(sourceFunctionType.getParams())) {
    const bool pointer =
        std::holds_alternative<sim::StructuredRuntimeMemoryInput>(
            structuredWorkload->argumentPlan[indexed.index()]);
    const auto integer = mlir::dyn_cast<mlir::IntegerType>(indexed.value());
    if (pointer != mlir::isa<mlir::LLVM::LLVMPointerType>(indexed.value()) ||
        (!pointer && (!integer || integer.getWidth() != 64)))
      return mismatch("product source argument type differs from its ABI");
  }
  for (std::size_t input = 0; input != cachedInputCount; ++input) {
    if (!std::holds_alternative<sim::StructuredRuntimeMemoryInput>(
            structuredWorkload->argumentPlan[input * 2]))
      return mismatch(
          "product cached input pointer is outside its ABI position");
    const std::optional<std::uint64_t> byteCount =
        definedU64(structuredWorkload->argumentPlan[input * 2 + 1]);
    if (!byteCount || *byteCount == 0)
      return mismatch("product cached input has no fixed positive byte extent");
  }
  const std::size_t profileBase = cachedInputCount * 2;
  const std::optional<std::uint64_t> warmup =
      definedU64(structuredWorkload->argumentPlan[profileBase]);
  const std::optional<std::uint64_t> measured =
      definedU64(structuredWorkload->argumentPlan[profileBase + 1]);
  const std::optional<std::uint64_t> outputBytes =
      definedU64(structuredWorkload->argumentPlan[profileBase + 3]);
  if (!warmup || *warmup != contract.warmupSamples || !measured ||
      *measured != contract.measuredSamples || !outputBytes ||
      *outputBytes != expectedOutputBytes ||
      !std::holds_alternative<sim::StructuredRuntimeMemoryInput>(
          structuredWorkload->argumentPlan[profileBase + 2]))
    return mismatch("product profile arguments differ from the contract");
  if (!structuredRuntime->runtimeValues.empty() ||
      structuredRuntime->pointerBindings.size() != pointerCount ||
      structuredRuntime->memoryObjects.size() != pointerCount)
    return mismatch(
        "product runtime input is not a total unaliased memory binding");

  const auto *systemWorkload = activationInputs.workload.system();
  const auto *systemRuntime = activationInputs.runtimeInput.system();
  if (!systemWorkload || !systemRuntime ||
      systemRuntime->memoryInterfaceBindings.size() != pointerCount ||
      systemRuntime->memoryObjects.size() != pointerCount)
    return mismatch(
        "Deployment activation does not preserve the product memory table");
  std::vector<bool> sourceObjectsUsed(pointerCount, false);
  std::vector<bool> systemObjectsUsed(pointerCount, false);
  std::uint64_t pointerOrdinal = 0;
  for (const auto indexed :
       llvm::enumerate(structuredWorkload->argumentPlan)) {
    if (!std::holds_alternative<sim::StructuredRuntimeMemoryInput>(
            indexed.value()))
      continue;
    const auto sourceBinding = llvm::find_if(
        structuredRuntime->pointerBindings,
        [&](const sim::StructuredPointerBindingEntry &binding) {
          return binding.argumentOrdinal == indexed.index();
        });
    const deployment::DeploymentExternalInterfaceRef interfaceRef{
        deployment.reference().artifact, pointerOrdinal};
    const auto systemBinding = llvm::find_if(
        systemRuntime->memoryInterfaceBindings,
        [&](const sim::SystemMemoryInterfaceBindingEntry &binding) {
          return binding.interfaceRef == interfaceRef;
        });
    if (sourceBinding == structuredRuntime->pointerBindings.end() ||
        systemBinding == systemRuntime->memoryInterfaceBindings.end() ||
        sourceBinding->binding.byteOffset != 0 ||
        systemBinding->binding.byteOffset != 0 ||
        sourceBinding->binding.objectOrdinal !=
            systemBinding->binding.objectOrdinal ||
        sourceBinding->binding.objectOrdinal >= pointerCount ||
        systemBinding->binding.objectOrdinal >= pointerCount ||
        sourceObjectsUsed[sourceBinding->binding.objectOrdinal] ||
        systemObjectsUsed[systemBinding->binding.objectOrdinal])
      return mismatch(
          "product memory binding is aliased, partial, or noncanonical");
    sourceObjectsUsed[sourceBinding->binding.objectOrdinal] = true;
    systemObjectsUsed[systemBinding->binding.objectOrdinal] = true;
    const sim::RuntimeMemoryObject &sourceObject =
        structuredRuntime->memoryObjects[sourceBinding->binding.objectOrdinal];
    const sim::RuntimeMemoryObject &systemObject =
        systemRuntime->memoryObjects[systemBinding->binding.objectOrdinal];
    const std::optional<std::uint64_t> declaredBytes =
        definedU64(structuredWorkload->argumentPlan[indexed.index() + 1]);
    if (!declaredBytes ||
        *declaredBytes != sourceObject.initialBytes.size() ||
        sourceObject.initialBytes.size() != systemObject.initialBytes.size() ||
        !sourceObject.pointerValues.empty() ||
        !systemObject.pointerValues.empty() ||
        !llvm::equal(
            sourceObject.initialBytes, systemObject.initialBytes,
            [](const sim::SemanticMemoryByte &sourceByte,
               const sim::SemanticMemoryByte &systemByte) {
              return sourceByte.state == systemByte.state &&
                     (sourceByte.state != sim::SemanticState::Defined ||
                      sourceByte.value == systemByte.value);
            }))
      return mismatch("Deployment activation changed product memory input "
                      "bytes");
    ++pointerOrdinal;
  }

  const std::uint64_t outputArgument = profileBase + 2;
  const auto outputBinding = llvm::find_if(
      structuredRuntime->pointerBindings,
      [&](const sim::StructuredPointerBindingEntry &binding) {
        return binding.argumentOrdinal == outputArgument;
      });
  if (outputBinding == structuredRuntime->pointerBindings.end() ||
      outputBinding->binding.byteOffset != 0 ||
      outputBinding->binding.objectOrdinal >=
          structuredRuntime->memoryObjects.size())
    return mismatch("product output pointer has no whole-object binding");
  const sim::RuntimeMemoryObject &sourceOutput =
      structuredRuntime->memoryObjects[outputBinding->binding.objectOrdinal];
  if (sourceOutput.initialBytes.size() != expectedOutputBytes ||
      !sourceOutput.pointerValues.empty() ||
      llvm::any_of(sourceOutput.initialBytes,
                   [](const sim::SemanticMemoryByte &byte) {
                     return byte.state != sim::SemanticState::Defined ||
                            byte.value != 0;
                   }))
    return mismatch("product output buffer is not independent zeroed storage");
  if (structuredWorkload->observableContract.memories.size() != 1 ||
      structuredWorkload->observableContract.memories.front().form !=
          sim::MemoryObservationForm::FullState)
    return mismatch("product output is not the sole full-state memory "
                    "observable");
  const auto *sourceObservable = std::get_if<sim::EntryPointerArgumentTarget>(
      &structuredWorkload->observableContract.memories.front().target);
  if (!sourceObservable ||
      sourceObservable->argumentOrdinal != outputArgument)
    return mismatch("product memory observable does not select the output "
                    "pointer");

  if (systemWorkload->valueInputPlan.size() != cachedInputCount + 3 ||
      !systemRuntime->runtimeEntryValues.empty() ||
      systemWorkload->observableContract.valueResults !=
          std::vector<std::uint64_t>{0} ||
      systemWorkload->observableContract.memories.size() != 1)
    return mismatch("Deployment activation does not expose the product ABI");
  std::vector<std::uint64_t> expectedValues;
  expectedValues.reserve(cachedInputCount + 3);
  for (const sim::StructuredProgramArgumentSource &argument :
       structuredWorkload->argumentPlan) {
    if (std::holds_alternative<sim::StructuredRuntimeMemoryInput>(argument))
      continue;
    const std::optional<std::uint64_t> value = definedU64(argument);
    if (!value)
      return mismatch("product source profile contains a non-u64 value");
    expectedValues.push_back(*value);
  }
  for (const auto indexed : llvm::enumerate(systemWorkload->valueInputPlan)) {
    const auto *fixed =
        std::get_if<sim::CanonicalValueSequence>(&indexed.value());
    const std::optional<std::uint64_t> value =
        fixed ? definedU64(*fixed) : std::nullopt;
    if (!value || *value != expectedValues[indexed.index()])
      return mismatch(
          "Deployment activation value differs from the source profile");
  }
  const sim::SystemMemoryObservable &systemObservable =
      systemWorkload->observableContract.memories.front();
  if (systemObservable.interfaceRef.externalInterfaceOrdinal !=
          contract.outputInterfaceOrdinal ||
      systemObservable.form != sim::MemoryObservationForm::FullState)
    return mismatch("Deployment activation observes another product output");
  const auto systemOutputBinding = llvm::find_if(
      systemRuntime->memoryInterfaceBindings,
      [&](const sim::SystemMemoryInterfaceBindingEntry &binding) {
        return binding.interfaceRef == systemObservable.interfaceRef;
      });
  if (systemOutputBinding == systemRuntime->memoryInterfaceBindings.end() ||
      systemOutputBinding->binding.byteOffset != 0 ||
      systemOutputBinding->binding.objectOrdinal >=
          systemRuntime->memoryObjects.size())
    return mismatch("Deployment activation output has no whole-object "
                    "binding");
  const sim::RuntimeMemoryObject &systemOutput =
      systemRuntime
          ->memoryObjects[systemOutputBinding->binding.objectOrdinal];
  if (systemOutput.initialBytes.size() != expectedOutputBytes ||
      !systemOutput.pointerValues.empty() ||
      llvm::any_of(systemOutput.initialBytes,
                   [](const sim::SemanticMemoryByte &byte) {
                     return byte.state != sim::SemanticState::Defined ||
                            byte.value != 0;
                   }))
    return mismatch("Deployment activation output binding is not independent "
                    "zeroed storage");

  auto hostEntry = deployment::resolveDeploymentProgramEntry(
      deployment, systemWorkload->programEntryRef);
  if (!hostEntry)
    return mismatch("product host entry cannot be resolved: " +
                    llvm::toString(hostEntry.takeError()));
  const deployment::HostProgramLeaf &hostProgram =
      deployment.deployment().hostProgram();
  const auto hostInterfaces = hostProgram.externalInterfaces();
  if (hostProgram.programEntries().size() != 1 ||
      (*hostEntry)->externalInterfaceOrdinals.size() != pointerCount ||
      hostInterfaces.size() != pointerCount)
    return mismatch(
        "product host memory-interface catalog differs from its ABI");
  auto expectedMemoryType = dataflow::encodeCanonicalType(mlir::MemRefType::get(
      {mlir::ShapedType::kDynamic},
      mlir::IntegerType::get(sourceFunction.getContext(), 8)));
  if (!expectedMemoryType)
    return mismatch("product byte-buffer type cannot be encoded: " +
                    llvm::toString(expectedMemoryType.takeError()));
  for (std::size_t ordinal = 0; ordinal != pointerCount; ++ordinal) {
    const deployment::HostExternalInterface &interface =
        hostInterfaces[ordinal];
    const deployment::HostExternalInterfaceDirection expectedDirection =
        ordinal == contract.outputInterfaceOrdinal
            ? deployment::HostExternalInterfaceDirection::Output
            : deployment::HostExternalInterfaceDirection::Input;
    if ((*hostEntry)->externalInterfaceOrdinals[ordinal] != ordinal ||
        interface.interfaceOrdinal != ordinal ||
        interface.kind != deployment::HostExternalInterfaceKind::Memory ||
        interface.direction != expectedDirection ||
        !llvm::equal(interface.semanticType, expectedMemoryType->bytes()))
      return mismatch(
          "product host memory interface differs from its byte-buffer ABI");
  }
  auto shapes = sim::projectSystemSimulationBoundaryShapes(
      deployment, systemWorkload->programEntryRef, artifacts);
  if (!shapes)
    return mismatch("product Deployment ABI cannot be projected: " +
                    llvm::toString(shapes.takeError()));
  if (!shapes->littleEndian ||
      shapes->valueArguments.size() != cachedInputCount + 3 ||
      llvm::any_of(shapes->valueArguments, [](const auto &shape) {
        return shape.pointerPayload || shape.lanesPerToken != 1 ||
               shape.laneBitWidth != 64;
      }) ||
      shapes->valueResults.size() != 1 ||
      shapes->valueResults.front().pointerPayload ||
      shapes->valueResults.front().lanesPerToken != 1 ||
      shapes->valueResults.front().laneBitWidth != 32)
    return mismatch("Deployment program entry differs from the product ABI");
  return llvm::Error::success();
}

} // namespace loom::application::detail
