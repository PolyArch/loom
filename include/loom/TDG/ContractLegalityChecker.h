#ifndef LOOM_TDG_CONTRACTLEGALITYCHECKER_H
#define LOOM_TDG_CONTRACTLEGALITYCHECKER_H

#include "mlir/IR/BuiltinOps.h"
#include <string>
#include <vector>

// Forward declarations for types from other plans (P05)
namespace loom {
struct SystemArchitecture;
struct AssignmentResult;
} // namespace loom

namespace loom {

/// Describes a single legality violation found in a contract.
struct LegalityViolation {
  /// The type of legality condition that was violated.
  enum Condition {
    BUFFER_CAPACITY,      // buffer < min_buffer_elements
    NOC_BANDWIDTH,        // bandwidth insufficient for rate
    SPM_OVERFLOW,         // SPM allocation exceeds budget
    ORDERING_VIOLATION,   // reorder permission inconsistent with ordering
    TYPE_INCOMPATIBILITY, // producer/consumer type mismatch
  };

  Condition condition;
  std::string contractName;
  std::string details;
};

/// Checks the 5 legality conditions from the contract specification
/// for all contracts in a TDG module.
///
/// Some checks are purely structural (can be done without architecture
/// information), while others require knowledge of the target
/// architecture and kernel-to-core assignment.
class ContractLegalityChecker {
public:
  /// Check all 5 legality conditions for all contracts in a TDG.
  ///
  /// The arch and assignment parameters may be null for structural-only
  /// checks (ordering violations and type incompatibilities). When both
  /// are provided, all 5 conditions are checked.
  std::vector<LegalityViolation>
  check(mlir::ModuleOp tdgModule,
        const SystemArchitecture *arch = nullptr,
        const AssignmentResult *assignment = nullptr);

  /// Check only structural legality (no architecture needed).
  std::vector<LegalityViolation>
  checkStructural(mlir::ModuleOp tdgModule);

private:
  /// Check ordering/reorder consistency.
  void checkOrderingConsistency(
      mlir::Operation *contractOp,
      std::vector<LegalityViolation> &violations);

  /// Check type compatibility between producer and consumer.
  void checkTypeCompatibility(
      mlir::Operation *contractOp,
      std::vector<LegalityViolation> &violations);

  /// Check buffer capacity constraints.
  /// Requires architecture and assignment information.
  void checkBufferCapacity(
      mlir::Operation *contractOp,
      const SystemArchitecture *arch,
      const AssignmentResult *assignment,
      std::vector<LegalityViolation> &violations);

  /// Check NoC bandwidth constraints.
  /// Requires architecture and assignment information.
  void checkNoCBandwidth(
      mlir::Operation *contractOp,
      const SystemArchitecture *arch,
      const AssignmentResult *assignment,
      std::vector<LegalityViolation> &violations);

  /// Check SPM overflow constraints.
  /// Requires architecture and assignment information.
  void checkSPMOverflow(
      mlir::Operation *contractOp,
      const SystemArchitecture *arch,
      const AssignmentResult *assignment,
      std::vector<LegalityViolation> &violations);
};

} // namespace loom

#endif
