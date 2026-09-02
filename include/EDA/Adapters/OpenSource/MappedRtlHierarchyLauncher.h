#ifndef LOOM_EDA_ADAPTERS_OPENSOURCE_MAPPEDRTLHIERARCHYLAUNCHER_H
#define LOOM_EDA_ADAPTERS_OPENSOURCE_MAPPEDRTLHIERARCHYLAUNCHER_H

#include "llvm/ADT/StringRef.h"

namespace loom::eda::open_source {

/// Contract between the mapped-RTL adapters and the hierarchy Verilator
/// launcher, the typed auxiliary tool that the Verilator-generated hierarchy
/// makefile invokes in place of Verilator for every child and root
/// Verilation. Verilator propagates every explicit SystemVerilog input of the
/// planning command into each child argument file, so every child would
/// otherwise elaborate the complete design through the harness top. The
/// launcher removes the harness token from child argument files only; it
/// never reads SystemVerilog and never edits a Verilator-generated file in
/// place.
///
/// The adapters freeze the launcher in the manifest's auxiliary-tool domain
/// under `mappedRtlHierarchyLauncherSlot` and pass the frozen Verilator
/// executable and the bundle-relative harness path as make command-line
/// variables, which GNU make exports to the launcher's environment.
inline constexpr llvm::StringLiteral mappedRtlHierarchyLauncherSlot =
    "mapped_rtl_hierarchy_launcher";
inline constexpr llvm::StringLiteral mappedRtlHierarchyVerilatorVariable =
    "LOOM_MAPPED_RTL_HIERARCHY_VERILATOR";
inline constexpr llvm::StringLiteral mappedRtlHierarchyTestbenchVariable =
    "LOOM_MAPPED_RTL_HIERARCHY_TESTBENCH";
/// Verilator's own make variable naming the Verilator executable that the
/// generated hierarchy makefile launches.
inline constexpr llvm::StringLiteral verilatorHierarchyLauncherVariable =
    "VM_HIER_VERILATOR";
/// Verilator's internal option marking a child argument file.
inline constexpr llvm::StringLiteral verilatorHierarchicalChildOption =
    "--hierarchical-child";
/// Suffix appended to a Verilator-generated child argument file name to form
/// the immutable filtered sibling that the launcher hands to Verilator.
inline constexpr llvm::StringLiteral mappedRtlHierarchyChildArgumentsSuffix =
    ".loom-hierarchy-child.f";

/// Launcher-authored exit codes. Verilator keeps its native exit code when it
/// is reached.
enum class MappedRtlHierarchyLauncherExit : int {
  /// A child argument file did not contain the harness token exactly once.
  TestbenchTokenCount = 42,
  /// The argument file could not be read, the filtered sibling could not be
  /// published, or Verilator could not be executed.
  InputOutput = 43,
  /// The make-exported configuration variables are absent or empty.
  Configuration = 44,
};

} // namespace loom::eda::open_source

#endif // LOOM_EDA_ADAPTERS_OPENSOURCE_MAPPEDRTLHIERARCHYLAUNCHER_H
