#ifndef LOOM_MAPPING_ARTIFACT_MAPPINGCONSTRAINTCANONICALIZATION_H
#define LOOM_MAPPING_ARTIFACT_MAPPINGCONSTRAINTCANONICALIZATION_H

#include "Mapping/IR/MappingOps.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLFunctionalExtras.h"

#include <cstdint>
#include <string>
#include <vector>

namespace loom::mapping::detail {

using ConstraintDomainTransform =
    llvm::function_ref<std::vector<mlir::Attribute>(
        mlir::MLIRContext *, mlir::Attribute, llvm::ArrayRef<mlir::Attribute>)>;

using ConstraintDomainIntersection =
    llvm::function_ref<std::vector<mlir::Attribute>(
        mlir::MLIRContext *, mlir::Attribute, llvm::ArrayRef<mlir::Attribute>,
        llvm::ArrayRef<mlir::Attribute>)>;

std::string constraintAttributeKey(mlir::Attribute attribute);

std::vector<mlir::Attribute>
normalizeExactConstraintDomain(llvm::ArrayRef<mlir::Attribute> values);

std::vector<mlir::Attribute>
intersectExactConstraintDomains(llvm::ArrayRef<mlir::Attribute> lhs,
                                llvm::ArrayRef<mlir::Attribute> rhs);

std::vector<mlir::Attribute> normalizeUnsignedIntervalConstraintDomain(
    mlir::MLIRContext *context, llvm::ArrayRef<mlir::Attribute> values);

std::vector<mlir::Attribute>
intersectUnsignedIntervalConstraintDomains(mlir::MLIRContext *context,
                                           llvm::ArrayRef<mlir::Attribute> lhs,
                                           llvm::ArrayRef<mlir::Attribute> rhs);

void canonicalizeConstraintClauses(
    mlir::Block &body, mlir::Location location,
    ConstraintDomainTransform normalizeDomain,
    ConstraintDomainIntersection intersectDomains);

} // namespace loom::mapping::detail

#endif // LOOM_MAPPING_ARTIFACT_MAPPINGCONSTRAINTCANONICALIZATION_H
