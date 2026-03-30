# Hierarchical compiler bilevel loop tests (D1)

add_executable(hierarchical-compiler-test HierarchicalCompilerTest.cpp)

target_include_directories(hierarchical-compiler-test PRIVATE
  ${LOOM_INCLUDE_DIR}
  ${CMAKE_BINARY_DIR}/include
)

llvm_map_components_to_libnames(HC_TEST_LLVM_LIBS core support)
llvm_map_components_to_libnames(HC_TEST_NATIVE_LLVM_LIBS
  native nativecodegen
  ${LLVM_TARGETS_TO_BUILD}
)

set(HC_TEST_RISCV_LINK_OPTIONS
  "LINKER:-u,LLVMInitializeRISCVTargetInfo"
  "LINKER:-u,LLVMInitializeRISCVTarget"
  "LINKER:-u,LLVMInitializeRISCVTargetMC"
)

target_link_libraries(hierarchical-compiler-test PRIVATE
  LoomSystemCompiler
  ${HC_TEST_LLVM_LIBS}
  ${HC_TEST_NATIVE_LLVM_LIBS}
)

set_target_properties(hierarchical-compiler-test PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${LOOM_RUNTIME_DIR}
)
target_link_options(hierarchical-compiler-test PRIVATE
  ${HC_TEST_RISCV_LINK_OPTIONS}
)

# TemporalScheduler and ADGPartitioner tests

add_executable(temporal-scheduler-test TemporalSchedulerTest.cpp)

target_include_directories(temporal-scheduler-test PRIVATE
  ${LOOM_INCLUDE_DIR}
  ${CMAKE_BINARY_DIR}/include
)

target_link_libraries(temporal-scheduler-test PRIVATE
  LoomSystemCompiler
  ${HC_TEST_LLVM_LIBS}
  ${HC_TEST_NATIVE_LLVM_LIBS}
)

set_target_properties(temporal-scheduler-test PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${LOOM_RUNTIME_DIR}
)
target_link_options(temporal-scheduler-test PRIVATE
  ${HC_TEST_RISCV_LINK_OPTIONS}
)

# NoC/Buffer/DMA integration tests (D3)

add_executable(noc-buffer-dma-test NoCBufferDMATest.cpp)

target_include_directories(noc-buffer-dma-test PRIVATE
  ${LOOM_INCLUDE_DIR}
  ${CMAKE_BINARY_DIR}/include
)

target_link_libraries(noc-buffer-dma-test PRIVATE
  LoomSystemCompiler
  ${HC_TEST_LLVM_LIBS}
  ${HC_TEST_NATIVE_LLVM_LIBS}
)

set_target_properties(noc-buffer-dma-test PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${LOOM_RUNTIME_DIR}
)
target_link_options(noc-buffer-dma-test PRIVATE
  ${HC_TEST_RISCV_LINK_OPTIONS}
)

# RTL generation wiring tests

add_executable(rtlgen-test RtlGenTest.cpp)

target_include_directories(rtlgen-test PRIVATE
  ${LOOM_INCLUDE_DIR}
  ${CMAKE_BINARY_DIR}/include
)

target_link_libraries(rtlgen-test PRIVATE
  LoomSystemCompiler
  LoomSVGen
  LoomADG
  LoomFabric
  LoomDataflow

  CIRCTHandshake

  MLIRIR
  MLIRParser
  MLIRSupport
  MLIRArithDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRFuncDialect
  MLIRSCFDialect

  ${HC_TEST_LLVM_LIBS}
  ${HC_TEST_NATIVE_LLVM_LIBS}
)

set_target_properties(rtlgen-test PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${LOOM_RUNTIME_DIR}
)
target_link_options(rtlgen-test PRIVATE
  ${HC_TEST_RISCV_LINK_OPTIONS}
)

# Function unit support coverage for rsqrt/minimumf across SVGen and simulator.

add_executable(svgen-function-unit-ops-test SVGenFunctionUnitOpsTest.cpp)

target_include_directories(svgen-function-unit-ops-test PRIVATE
  ${LOOM_INCLUDE_DIR}
  ${CMAKE_BINARY_DIR}/include
  ${LOOM_SOURCE_DIR}/lib/loom/SVGen
)

target_link_libraries(svgen-function-unit-ops-test PRIVATE
  LoomSVGen
  LoomSimulator
  LoomFabric

  MLIRIR
  MLIRSupport
  MLIRArithDialect
  MLIRMathDialect

  ${HC_TEST_LLVM_LIBS}
)

set_target_properties(svgen-function-unit-ops-test PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${LOOM_RUNTIME_DIR}
)

# LLVM-to-CF pointer/null compare regression test

add_executable(llvmto-cf-icmp-null-test LLVMToCFNullIcmpTest.cpp)

target_include_directories(llvmto-cf-icmp-null-test PRIVATE
  ${LOOM_INCLUDE_DIR}
  ${CMAKE_BINARY_DIR}/include
)

target_link_libraries(llvmto-cf-icmp-null-test PRIVATE
  LoomConversion
  MLIRArithDialect
  MLIRFuncDialect
  MLIRLLVMDialect
  MLIRMemRefDialect
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRSupport
  ${HC_TEST_LLVM_LIBS}
)

set_target_properties(llvmto-cf-icmp-null-test PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${LOOM_RUNTIME_DIR}
)

# SCF-to-DFG dominance regression test

add_executable(scf-to-dfg-dominance-test SCFToDFGDominanceTest.cpp)

target_include_directories(scf-to-dfg-dominance-test PRIVATE
  ${LOOM_INCLUDE_DIR}
  ${CMAKE_BINARY_DIR}/include
)

target_link_libraries(scf-to-dfg-dominance-test PRIVATE
  LoomConversion

  CIRCTHandshake

  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRMemRefDialect
  MLIRParser
  MLIRPass
  MLIRSCFDialect
  MLIRSupport

  ${HC_TEST_LLVM_LIBS}
)

set_target_properties(scf-to-dfg-dominance-test PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${LOOM_RUNTIME_DIR}
)

# SciComp KernelCompiler function-path regression test

add_executable(scicomp-kernelcompiler-test SciCompKernelCompilerTest.cpp)

target_include_directories(scicomp-kernelcompiler-test PRIVATE
  ${LOOM_INCLUDE_DIR}
  ${CMAKE_BINARY_DIR}/include
)

target_link_libraries(scicomp-kernelcompiler-test PRIVATE
  LoomSystemCompiler

  CIRCTHandshake

  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRDLTIDialect
  MLIRFuncDialect
  MLIRIR
  MLIRLLVMDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRParser
  MLIRPass
  MLIRSCFDialect
  MLIRSupport
  MLIRUBDialect

  ${HC_TEST_LLVM_LIBS}
  ${HC_TEST_NATIVE_LLVM_LIBS}
)

set_target_properties(scicomp-kernelcompiler-test PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${LOOM_RUNTIME_DIR}
)
target_link_options(scicomp-kernelcompiler-test PRIVATE
  ${HC_TEST_RISCV_LINK_OPTIONS}
)

# Scientific computing KHG/SHG mapping validation

add_executable(scicomp-mapping-test SciCompMappingTest.cpp)

target_include_directories(scicomp-mapping-test PRIVATE
  ${LOOM_INCLUDE_DIR}
  ${CMAKE_BINARY_DIR}/include
)

target_compile_definitions(scicomp-mapping-test PRIVATE
  LOOM_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
)

target_link_libraries(scicomp-mapping-test PRIVATE
  LoomSystemCompiler
  ${HC_TEST_LLVM_LIBS}
  ${HC_TEST_NATIVE_LLVM_LIBS}
)

set_target_properties(scicomp-mapping-test PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${LOOM_RUNTIME_DIR}
)
target_link_options(scicomp-mapping-test PRIVATE
  ${HC_TEST_RISCV_LINK_OPTIONS}
)
