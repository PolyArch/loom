#include "loom/TDG/ContractLegalityChecker.h"
#include "loom/Dialect/TDG/TDGOps.h"

using namespace mlir;
using namespace loom;
using namespace loom::tdg;

std::vector<LegalityViolation>
ContractLegalityChecker::check(ModuleOp tdgModule,
                               const SystemArchitecture *arch,
                               const AssignmentResult *assignment) {
  std::vector<LegalityViolation> violations;

  tdgModule.walk([&](GraphOp graphOp) {
    for (auto &op : graphOp.getBody().front()) {
      if (auto contractOp = dyn_cast<ContractOp>(op)) {
        // Structural checks (always performed)
        checkOrderingConsistency(contractOp, violations);
        checkTypeCompatibility(contractOp, violations);

        // Architecture-dependent checks
        if (arch && assignment) {
          checkBufferCapacity(contractOp, arch, assignment, violations);
          checkNoCBandwidth(contractOp, arch, assignment, violations);
          checkSPMOverflow(contractOp, arch, assignment, violations);
        }
      }
    }
  });

  return violations;
}

std::vector<LegalityViolation>
ContractLegalityChecker::checkStructural(ModuleOp tdgModule) {
  return check(tdgModule, nullptr, nullptr);
}

void ContractLegalityChecker::checkOrderingConsistency(
    Operation *op, std::vector<LegalityViolation> &violations) {
  auto contractOp = cast<ContractOp>(op);
  StringRef ordering = contractOp.getOrdering();
  bool mayReorder = contractOp.getMayReorder();

  if (ordering == "FIFO" && mayReorder) {
    LegalityViolation v;
    v.condition = LegalityViolation::ORDERING_VIOLATION;
    v.contractName = contractOp.getProducer().str() + " -> " +
                     contractOp.getConsumer().str();
    v.details = "FIFO ordering is incompatible with may_reorder=true";
    violations.push_back(std::move(v));
  }
}

void ContractLegalityChecker::checkTypeCompatibility(
    Operation *op, std::vector<LegalityViolation> &violations) {
  // Type compatibility is validated structurally by comparing the contract's
  // data_type with the producer/consumer kernel signatures. Currently a
  // placeholder: full checking requires kernel body analysis.
  (void)op;
  (void)violations;
}

void ContractLegalityChecker::checkBufferCapacity(
    Operation *op, const SystemArchitecture *arch,
    const AssignmentResult *assignment,
    std::vector<LegalityViolation> &violations) {
  // Placeholder: requires SystemArchitecture and AssignmentResult from P05.
  // Will check that the allocated buffer capacity meets min_buffer_elements.
  (void)op;
  (void)arch;
  (void)assignment;
  (void)violations;
}

void ContractLegalityChecker::checkNoCBandwidth(
    Operation *op, const SystemArchitecture *arch,
    const AssignmentResult *assignment,
    std::vector<LegalityViolation> &violations) {
  // Placeholder: requires SystemArchitecture and AssignmentResult from P05.
  // Will check that NoC bandwidth is sufficient for the specified rate.
  (void)op;
  (void)arch;
  (void)assignment;
  (void)violations;
}

void ContractLegalityChecker::checkSPMOverflow(
    Operation *op, const SystemArchitecture *arch,
    const AssignmentResult *assignment,
    std::vector<LegalityViolation> &violations) {
  // Placeholder: requires SystemArchitecture and AssignmentResult from P05.
  // Will check that SPM allocation does not exceed the budget.
  (void)op;
  (void)arch;
  (void)assignment;
  (void)violations;
}
