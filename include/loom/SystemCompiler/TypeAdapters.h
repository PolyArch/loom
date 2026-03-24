//===-- TypeAdapters.h - Bridge between type namespaces -----------*- C++ -*-===//
//
// Adapter functions that convert between the three type ecosystems used in
// the Tapestry multi-core compiler:
//
//   1. loom::tapestry  -- MLIR-centric types (SystemArchitecture, KernelDesc)
//   2. loom (root)     -- abstract types (SystemArchitecture, KernelProfile)
//   3. loom::syscomp   -- legacy types (BendersTask, BendersResult)
//
// These adapters operate at namespace boundaries so that existing components
// (L1CoreAssigner, L2CoreCompiler, BendersHelpers, NoCScheduler) can be
// called without modifying their interfaces.
//
//===----------------------------------------------------------------------===//

#ifndef LOOM_SYSTEMCOMPILER_TYPEADAPTERS_H
#define LOOM_SYSTEMCOMPILER_TYPEADAPTERS_H

#include "loom/SystemCompiler/BendersHelpers.h"
#include "loom/SystemCompiler/Contract.h"
#include "loom/SystemCompiler/KernelProfiler.h"
#include "loom/SystemCompiler/L1CoreAssignment.h"
#include "loom/SystemCompiler/L2CoreCompiler.h"
#include "loom/SystemCompiler/SystemTypes.h"

#include "mlir/IR/BuiltinOps.h"

#include <map>
#include <string>
#include <vector>

namespace llvm {
namespace json {
class Value;
} // namespace json
} // namespace llvm

namespace loom {

/// Convert a tapestry::SystemArchitecture (MLIR ADG modules) to a
/// loom::SystemArchitecture (FU type counts) suitable for L1CoreAssigner.
SystemArchitecture
toL1Architecture(const tapestry::SystemArchitecture &tapArch,
                 mlir::MLIRContext *ctx);

/// Convert a tapestry::KernelDesc (MLIR DFG module) to a loom::KernelProfile
/// (operation counts) suitable for L1CoreAssigner.
KernelProfile toKernelProfile(const tapestry::KernelDesc &kernelDesc,
                              mlir::MLIRContext *ctx);

/// Profile all kernels and return a vector of KernelProfiles.
std::vector<KernelProfile>
toKernelProfiles(const std::vector<tapestry::KernelDesc> &kernels,
                 mlir::MLIRContext *ctx);

/// Convert tapestry::ContractSpec (with dataType, elementCount) to
/// loom::ContractSpec (with dataTypeName, rates, tile shapes).
ContractSpec toL1Contract(const tapestry::ContractSpec &tapContract);

/// Convert a vector of tapestry contracts to loom contracts.
std::vector<ContractSpec>
toL1Contracts(const std::vector<tapestry::ContractSpec> &tapContracts);

/// Build a kernel DFG lookup map from tapestry::KernelDesc entries.
std::map<std::string, mlir::ModuleOp>
buildKernelDFGMap(const std::vector<tapestry::KernelDesc> &kernels);

/// Look up the ADG module for a core type by name from the tapestry
/// architecture.
mlir::ModuleOp findCoreADG(const tapestry::SystemArchitecture &tapArch,
                           const std::string &coreTypeName);

/// Populate the coreADG field on L2Assignments using the tapestry architecture.
void populateL2ADGs(std::vector<L2Assignment> &l2Assignments,
                    const tapestry::SystemArchitecture &tapArch);

/// Convert L2Results and L1 assignment into a tapestry::BendersResult for
/// returning from the tapestry::BendersDriver.
tapestry::BendersResult
toBendersResult(const TapestryCompilationResult &compResult,
                const std::vector<tapestry::KernelDesc> &kernels,
                const tapestry::SystemArchitecture &arch,
                unsigned iterations);

/// Serialize a TapestryCompilationResult as JSON and write to the given path.
/// Returns true on success.
bool serializeResultJSON(const TapestryCompilationResult &result,
                         unsigned iterations,
                         double compilationTimeSec,
                         const std::string &outputPath);

} // namespace loom

#endif // LOOM_SYSTEMCOMPILER_TYPEADAPTERS_H
