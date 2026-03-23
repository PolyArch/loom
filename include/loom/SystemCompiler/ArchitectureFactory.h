//===-- ArchitectureFactory.h - Build SystemArchitectures ----------*- C++ -*-===//
//
// Factory utilities for constructing SystemArchitecture instances with real
// ADG modules built via the ADGBuilder.
//
//===----------------------------------------------------------------------===//

#ifndef LOOM_SYSTEMCOMPILER_ARCHITECTUREFACTORY_H
#define LOOM_SYSTEMCOMPILER_ARCHITECTUREFACTORY_H

#include "loom/SystemCompiler/SystemTypes.h"
#include "mlir/IR/MLIRContext.h"

#include <string>
#include <vector>

namespace loom {
namespace tapestry {

/// Specification for a core type to be built by the factory.
struct CoreTypeSpec {
  std::string name = "core";
  unsigned meshRows = 2;
  unsigned meshCols = 2;
  unsigned peBitsWidth = 32;
  unsigned numInstances = 1;
  unsigned spmSizeBytes = 4096;

  /// If true, add multiplier FUs to the PE definition.
  bool includeMultiplier = true;

  /// If true, add comparison FUs to the PE definition.
  bool includeComparison = true;

  /// If true, add memory load/store FUs and an external memory.
  bool includeMemory = true;
};

/// Build a SystemArchitecture from a list of core type specifications.
/// Each core type gets its own ADG module built via the ADGBuilder.
/// The ADG modules are parsed into the given MLIRContext.
///
/// Returns a populated SystemArchitecture on success.
/// On failure, returns a SystemArchitecture with an empty coreTypes vector.
SystemArchitecture buildArchitecture(const std::string &systemName,
                                     const std::vector<CoreTypeSpec> &specs,
                                     mlir::MLIRContext &ctx);

/// Build a standard general-purpose architecture with the given number of
/// core types and instances per type.
SystemArchitecture buildStandardArchitecture(const std::string &systemName,
                                             unsigned numCoreTypes,
                                             unsigned instancesPerType,
                                             unsigned meshRows,
                                             unsigned meshCols,
                                             mlir::MLIRContext &ctx);

} // namespace tapestry
} // namespace loom

#endif // LOOM_SYSTEMCOMPILER_ARCHITECTUREFACTORY_H
