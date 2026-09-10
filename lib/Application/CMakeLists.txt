find_package(Python3 REQUIRED COMPONENTS Interpreter)

add_library(LoomApplicationPortfolio
  Manifest.cpp
  SourceAdmission.cpp
)

target_include_directories(LoomApplicationPortfolio PUBLIC
  $<BUILD_INTERFACE:${LOOM_SOURCE_DIR}/include>
)

target_link_libraries(LoomApplicationPortfolio PUBLIC
  LoomCommon
  LLVMSupport
)

add_library(LoomApplicationHostRunner
  HostRunner.cpp
)

target_include_directories(LoomApplicationHostRunner PUBLIC
  $<BUILD_INTERFACE:${LOOM_SOURCE_DIR}/include>
)

target_link_libraries(LoomApplicationHostRunner PUBLIC
  LoomApplicationPortfolio
  LLVMSupport
)

add_library(LoomApplicationRuntimeManifest
  ActivationInputs.cpp
  ActivationDecision.cpp
  ApplicationRuntimeEvidenceJoin.cpp
  ApplicationSystemRuntimeEvidence.cpp
  ApplicationSystemRuntimeEvidenceJoin.cpp
  ActivationRepairLineage.cpp
  Package.cpp
  ProductOracleEvaluation.cpp
  RuntimeManifest.cpp
  RuntimeManifestValidation.cpp
  RuntimeProductContract.cpp
  SystemQor.cpp
)

target_include_directories(LoomApplicationRuntimeManifest PUBLIC
  $<BUILD_INTERFACE:${LOOM_SOURCE_DIR}/include>
)

target_link_libraries(LoomApplicationRuntimeManifest PUBLIC
  LoomCommon
  LoomDeployment
  LoomDSE
  LoomDSECompilerGenerator
  LoomDSEFeedback
  LoomEvaluation
  LoomFabricArtifact
  LoomFrontendCompilation
  LoomMappingArtifact
  LoomPnR
  LoomSimulator
  LoomStructuredFabricAnalyticModel
  LLVMSupport
  LoomGem5SystemExecution
)

file(GLOB_RECURSE LOOM_LLVM_LIBC_RUNTIME_HEADERS CONFIGURE_DEPENDS
  "${LOOM_EXTERNAL_SOURCE_DIR}/llvm/libc/hdr/*.h"
  "${LOOM_EXTERNAL_SOURCE_DIR}/llvm/libc/src/__support/*.h"
  "${LOOM_EXTERNAL_SOURCE_DIR}/llvm/libc/src/string/*.h"
)
set(LOOM_FREESTANDING_RUNTIME_COMPILE_OPTIONS
    --target=riscv64-unknown-elf
    -march=rv64imafdc_zicsr_zifencei
    -mabi=lp64d
    -mcmodel=medany
    -mcpu=generic-rv64
    -std=c++17
    -O2
    -ffreestanding
    -fno-exceptions
    -fno-rtti
    -DLIBC_NAMESPACE=loom_libc
    -DLIBC_FULL_BUILD
    -DLIBC_COPT_PUBLIC_PACKAGING
    -I${LOOM_EXTERNAL_SOURCE_DIR}/llvm/libc
    -emit-llvm
)
set(LOOM_FREESTANDING_MATH_BITCODE
  "${CMAKE_CURRENT_BINARY_DIR}/LoomFreestandingMath.bc")
set(LOOM_FREESTANDING_MATH_HEADER
  "${CMAKE_CURRENT_BINARY_DIR}/LoomFreestandingMathBitcode.h")
add_custom_command(
  OUTPUT "${LOOM_FREESTANDING_MATH_BITCODE}"
  COMMAND "${LLVM_TOOLS_BINARY_DIR}/clang++"
    ${LOOM_FREESTANDING_RUNTIME_COMPILE_OPTIONS}
    -fno-math-errno
    -DLIBC_MATH_HAS_ASSUME_ROUND_NEAREST_ONLY
    -DLIBC_MATH_HAS_NO_ERRNO
    -DLIBC_MATH_HAS_NO_EXCEPT
    -c "${LOOM_EXTERNAL_SOURCE_DIR}/llvm/libc/src/math/generic/expf.cpp"
    -o "${LOOM_FREESTANDING_MATH_BITCODE}"
  DEPENDS
    "${LOOM_EXTERNAL_SOURCE_DIR}/llvm/libc/src/math/generic/expf.cpp"
    ${LOOM_LLVM_LIBC_RUNTIME_HEADERS}
  VERBATIM
)
add_custom_command(
  OUTPUT "${LOOM_FREESTANDING_MATH_HEADER}"
  COMMAND "${Python3_EXECUTABLE}"
    "${LOOM_SOURCE_DIR}/scripts/embed_binary.py"
    --input "${LOOM_FREESTANDING_MATH_BITCODE}"
    --output "${LOOM_FREESTANDING_MATH_HEADER}"
    --symbol freestandingMathBitcode
  DEPENDS
    "${LOOM_FREESTANDING_MATH_BITCODE}"
    "${LOOM_SOURCE_DIR}/scripts/embed_binary.py"
  VERBATIM
)

set(LOOM_FREESTANDING_MEMORY_PARTS)
foreach(operation IN ITEMS memcpy memmove memset)
  set(part "${CMAKE_CURRENT_BINARY_DIR}/LoomFreestanding_${operation}.bc")
  add_custom_command(
    OUTPUT "${part}"
    COMMAND "${LLVM_TOOLS_BINARY_DIR}/clang++"
      ${LOOM_FREESTANDING_RUNTIME_COMPILE_OPTIONS}
      -c "${LOOM_EXTERNAL_SOURCE_DIR}/llvm/libc/src/string/${operation}.cpp"
      -o "${part}"
    DEPENDS
      "${LOOM_EXTERNAL_SOURCE_DIR}/llvm/libc/src/string/${operation}.cpp"
      ${LOOM_LLVM_LIBC_RUNTIME_HEADERS}
    VERBATIM
  )
  list(APPEND LOOM_FREESTANDING_MEMORY_PARTS "${part}")
endforeach()
set(LOOM_FREESTANDING_MEMORY_BITCODE
  "${CMAKE_CURRENT_BINARY_DIR}/LoomFreestandingMemory.bc")
set(LOOM_FREESTANDING_MEMORY_HEADER
  "${CMAKE_CURRENT_BINARY_DIR}/LoomFreestandingMemoryBitcode.h")
add_custom_command(
  OUTPUT "${LOOM_FREESTANDING_MEMORY_BITCODE}"
  COMMAND "${LLVM_TOOLS_BINARY_DIR}/llvm-link"
    ${LOOM_FREESTANDING_MEMORY_PARTS}
    -o "${LOOM_FREESTANDING_MEMORY_BITCODE}"
  DEPENDS ${LOOM_FREESTANDING_MEMORY_PARTS}
  VERBATIM
)
add_custom_command(
  OUTPUT "${LOOM_FREESTANDING_MEMORY_HEADER}"
  COMMAND "${Python3_EXECUTABLE}"
    "${LOOM_SOURCE_DIR}/scripts/embed_binary.py"
    --input "${LOOM_FREESTANDING_MEMORY_BITCODE}"
    --output "${LOOM_FREESTANDING_MEMORY_HEADER}"
    --symbol freestandingMemoryBitcode
  DEPENDS
    "${LOOM_FREESTANDING_MEMORY_BITCODE}"
    "${LOOM_SOURCE_DIR}/scripts/embed_binary.py"
  VERBATIM
)

add_library(LoomApplicationBuild
  ApplicationRuntimeValidation.cpp
  Build.cpp
  BuildDiagnostics.cpp
  PairDecisionDiagnostics.cpp
  DeploymentBuild.cpp
  ExecutionGlue.cpp
  SpatialInvocationPlan.cpp
  MappingExecution.cpp
  PairDecision.cpp
  ProductBuild.cpp
  ProductVisualization.cpp
  Quality.cpp
  SystemQuality.cpp
  "${LOOM_FREESTANDING_MATH_HEADER}"
  "${LOOM_FREESTANDING_MEMORY_HEADER}"
)

add_library(LoomApplicationDeploymentRuntime
  DeploymentRuntime.cpp
  ResourceTimeExecution.cpp
)

target_include_directories(LoomApplicationDeploymentRuntime PUBLIC
  $<BUILD_INTERFACE:${LOOM_SOURCE_DIR}/include>
)

target_link_libraries(LoomApplicationDeploymentRuntime PUBLIC
  LoomApplicationRuntimeManifest
  LoomCommon
  LoomDeployment
  LoomRuntimeResourceTime
  LoomSimulator
  LLVMSupport
)

target_include_directories(LoomApplicationBuild PUBLIC
  $<BUILD_INTERFACE:${LOOM_SOURCE_DIR}/include>
)
target_include_directories(LoomApplicationBuild PRIVATE
  "${CMAKE_CURRENT_BINARY_DIR}"
)

target_compile_definitions(LoomApplicationBuild PRIVATE
  LOOM_LLVM_TOOLS_DIR="${LLVM_TOOLS_BINARY_DIR}"
)

target_link_libraries(LoomApplicationBuild PUBLIC
  LoomApplicationRuntimeManifest
  LoomCommon
  LoomDeployment
  LoomDSE
  LoomDSECompilerGenerator
  LoomDSEProductionOwners
  LoomFabricArtifact
  LoomFrontendCompilation
  LoomFrontendExecutable
  LoomFrontendExecutableLinker
  LoomHardwareConfiguration
  LoomHardwareImplementation
  LoomMappingArtifact
  LoomRuntimePlatformBinding
  LoomGem5SystemExecution
  LoomSimulator
  LLVMCore
  LLVMipo
  LLVMTransformUtils
  LLVMSupport
)

target_link_libraries(LoomApplicationBuild PRIVATE
  LoomADGBuilder
  LoomApplicationDeploymentRuntime
  LoomApplicationPortfolio
  LoomExternalTool
)

mlir_target_link_libraries(LoomApplicationBuild PRIVATE
  MLIRArithToLLVM
  MLIRBuiltinToLLVMIRTranslation
  MLIRControlFlowToLLVM
  MLIRConvertToLLVM
  MLIRFuncToLLVM
  MLIRIndexToLLVM
  MLIRLLVMDialect
  MLIRLLVMToLLVMIRTranslation
  MLIRMathToLLVM
  MLIRMemRefToLLVM
  MLIRPass
  MLIRSCFToControlFlow
  MLIRTargetLLVMIRExport
  MLIRUBToLLVM
  MLIRVectorToLLVM
)
