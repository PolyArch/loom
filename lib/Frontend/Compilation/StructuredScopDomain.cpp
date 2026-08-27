#include "Frontend/Compilation/StructuredScop.h"

#include "llvm/Support/ErrorHandling.h"

namespace loom::frontend {

llvm::StringRef
structuredScopRefusalKindSpelling(StructuredScopRefusalKind kind) {
  switch (kind) {
  case StructuredScopRefusalKind::NotAffineLoop:
    return "not_affine_loop";
  case StructuredScopRefusalKind::NestedAffineRoot:
    return "nested_affine_root";
  case StructuredScopRefusalKind::NonCanonicalIterationDomain:
    return "noncanonical_iteration_domain";
  case StructuredScopRefusalKind::DomainProofNotEstablished:
    return "domain_proof_not_established";
  case StructuredScopRefusalKind::NestedControl:
    return "nested_control";
  case StructuredScopRefusalKind::UnsupportedEffect:
    return "unsupported_effect";
  case StructuredScopRefusalKind::UnsupportedOperation:
    return "unsupported_operation";
  case StructuredScopRefusalKind::AccessRelationProofNotEstablished:
    return "access_relation_proof_not_established";
  case StructuredScopRefusalKind::NonContiguousAccess:
    return "noncontiguous_access";
  case StructuredScopRefusalKind::AliasProofNotEstablished:
    return "alias_proof_not_established";
  case StructuredScopRefusalKind::DependenceProofNotEstablished:
    return "dependence_proof_not_established";
  case StructuredScopRefusalKind::LoopCarriedMemoryDependence:
    return "loop_carried_memory_dependence";
  case StructuredScopRefusalKind::AlignmentProofNotEstablished:
    return "alignment_proof_not_established";
  case StructuredScopRefusalKind::UnsupportedReduction:
    return "unsupported_reduction";
  case StructuredScopRefusalKind::StrictFloatingReduction:
    return "strict_floating_reduction";
  case StructuredScopRefusalKind::ProviderMaterializationRejected:
    return "provider_materialization_rejected";
  case StructuredScopRefusalKind::FabricCapabilityUnavailable:
    return "fabric_capability_unavailable";
  case StructuredScopRefusalKind::UnsupportedTail:
    return "unsupported_tail";
  case StructuredScopRefusalKind::NonUnitPhysicalStride:
    return "nonunit_physical_stride";
  case StructuredScopRefusalKind::HeterogeneousElementWidth:
    return "heterogeneous_element_width";
  case StructuredScopRefusalKind::IntegerOverflowReduction:
    return "integer_overflow_reduction";
  case StructuredScopRefusalKind::NonLocalMemoryRoot:
    return "nonlocal_memory_root";
  case StructuredScopRefusalKind::VectorLoweringUnavailable:
    return "vector_lowering_unavailable";
  case StructuredScopRefusalKind::UnsupportedPhysicalOffset:
    return "unsupported_physical_offset";
  case StructuredScopRefusalKind::ProviderDomainNotAdmitted:
    return "provider_domain_not_admitted";
  case StructuredScopRefusalKind::ProviderScheduleNotEstablished:
    return "provider_schedule_not_established";
  case StructuredScopRefusalKind::ProviderScheduleBudgetExhausted:
    return "provider_schedule_budget_exhausted";
  case StructuredScopRefusalKind::PolyhedralMaterializationUnavailable:
    return "polyhedral_materialization_unavailable";
  case StructuredScopRefusalKind::PhysicalLayoutProofNotEstablished:
    return "physical_layout_proof_not_established";
  }
  llvm_unreachable("unknown Structured SCoP refusal kind");
}

StructuredScopRefusalDisposition
classifyStructuredScopRefusal(StructuredScopRefusalKind kind) {
  switch (kind) {
  case StructuredScopRefusalKind::DomainProofNotEstablished:
  case StructuredScopRefusalKind::AccessRelationProofNotEstablished:
  case StructuredScopRefusalKind::AliasProofNotEstablished:
  case StructuredScopRefusalKind::DependenceProofNotEstablished:
  case StructuredScopRefusalKind::AlignmentProofNotEstablished:
  case StructuredScopRefusalKind::ProviderScheduleNotEstablished:
  case StructuredScopRefusalKind::ProviderScheduleBudgetExhausted:
  case StructuredScopRefusalKind::PolyhedralMaterializationUnavailable:
  case StructuredScopRefusalKind::PhysicalLayoutProofNotEstablished:
    return StructuredScopRefusalDisposition::IncompleteProof;
  case StructuredScopRefusalKind::NotAffineLoop:
  case StructuredScopRefusalKind::NestedAffineRoot:
  case StructuredScopRefusalKind::NonCanonicalIterationDomain:
  case StructuredScopRefusalKind::NestedControl:
  case StructuredScopRefusalKind::UnsupportedEffect:
  case StructuredScopRefusalKind::UnsupportedOperation:
  case StructuredScopRefusalKind::NonContiguousAccess:
  case StructuredScopRefusalKind::LoopCarriedMemoryDependence:
  case StructuredScopRefusalKind::UnsupportedReduction:
  case StructuredScopRefusalKind::StrictFloatingReduction:
  case StructuredScopRefusalKind::ProviderMaterializationRejected:
  case StructuredScopRefusalKind::FabricCapabilityUnavailable:
  case StructuredScopRefusalKind::UnsupportedTail:
  case StructuredScopRefusalKind::NonUnitPhysicalStride:
  case StructuredScopRefusalKind::HeterogeneousElementWidth:
  case StructuredScopRefusalKind::IntegerOverflowReduction:
  case StructuredScopRefusalKind::NonLocalMemoryRoot:
  case StructuredScopRefusalKind::VectorLoweringUnavailable:
  case StructuredScopRefusalKind::UnsupportedPhysicalOffset:
  case StructuredScopRefusalKind::ProviderDomainNotAdmitted:
    return StructuredScopRefusalDisposition::OutsideAdmittedDomain;
  }
  llvm_unreachable("unknown Structured SCoP refusal kind");
}

} // namespace loom::frontend
