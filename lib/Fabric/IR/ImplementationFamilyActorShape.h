#ifndef LOOM_FABRIC_IR_IMPLEMENTATION_FAMILY_ACTOR_SHAPE_H
#define LOOM_FABRIC_IR_IMPLEMENTATION_FAMILY_ACTOR_SHAPE_H

#include "Fabric/IR/ImplementationFamily.h"

// The capability-independent half of a typed admission provider: the ordered
// operand and result shape its actors must have before any capability domain
// is consulted. Forward admission proves the shape and then the capability;
// canonical capability derivation proves the shape, derives the least
// capability, and proves admission under it. Both read these one owners.
namespace fabric::detail {

llvm::Error
requireArity(const ::dataflow::CanonicalActorSchemaProjection &actor,
             unsigned inputs, unsigned results);
llvm::Error
requireUniformType(const ::dataflow::CanonicalActorSchemaProjection &actor,
                   unsigned inputs);

/// Whether one stream recurrence steps by a registered kind.
bool isValidStreamStepKind(::dataflow::StreamStepKind kind);

/// The rounding mode one arithmetic floating actor carries, defaulted to the
/// canonical round-to-nearest-even, or none when the actor has no floating
/// arithmetic projection at all.
std::optional<::mlir::arith::RoundingMode>
arithmeticRounding(const ::dataflow::CanonicalActorSchemaProjection &actor);

/// Which physical datapath a uniform floating family implements. The two admit
/// different schema inventories: only the scalar datapath carries the divide
/// and remainder resources.
enum class FloatDatapath : std::uint8_t { Scalar, FixedVector };

/// Ordered operand count of one uniform floating schema on one datapath, and
/// whether that schema rounds its result. A zero operand count means the
/// datapath's provider does not own the schema. Shape validation, admission,
/// and capability derivation all read this one table.
struct UniformFloatShape final {
  unsigned inputCount = 0;
  bool rounds = false;
};

UniformFloatShape uniformFloatShape(FloatDatapath datapath,
                                    ::dataflow::OperationSchemaId schema);

llvm::Error verifyScalarOrdinaryIntegerActorShape(
    const ::dataflow::CanonicalActorSchemaProjection &actor);
llvm::Error verifyScalarIntegerCastActorShape(
    const ::dataflow::CanonicalActorSchemaProjection &actor);
llvm::Error verifyScalarUniformFloatActorShape(
    const ::dataflow::CanonicalActorSchemaProjection &actor);
llvm::Error verifyFixedVectorUniformFloatActorShape(
    const ::dataflow::CanonicalActorSchemaProjection &actor);
llvm::Error
verifyStreamActorShape(const ::dataflow::CanonicalActorSchemaProjection &actor);
llvm::Error verifyTokenPlaneActorShape(
    const ::dataflow::CanonicalActorSchemaProjection &actor);
llvm::Error verifyConstantTokenActorShape(
    const ::dataflow::CanonicalActorSchemaProjection &actor);
llvm::Error verifySyncTokenActorShape(
    const ::dataflow::CanonicalActorSchemaProjection &actor);
llvm::Error verifyMuxTokenActorShape(
    const ::dataflow::CanonicalActorSchemaProjection &actor);
llvm::Error verifyDemuxTokenActorShape(
    const ::dataflow::CanonicalActorSchemaProjection &actor);

} // namespace fabric::detail

#endif // LOOM_FABRIC_IR_IMPLEMENTATION_FAMILY_ACTOR_SHAPE_H
