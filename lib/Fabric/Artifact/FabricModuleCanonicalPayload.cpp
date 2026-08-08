#include "FabricModuleCanonicalPayload.h"

#include "Dataflow/IR/OperationSchemaCodec.h"
#include "Fabric/IR/FabricCanonicalEntity.h"
#include "Fabric/IR/FabricOps.h"
#include "Fabric/IR/ResourceContractRecord.h"

#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Visitors.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

using namespace mlir;

namespace loom::fabric::detail {
namespace {

constexpr llvm::StringLiteral moduleAuthoringOnlyAttrs[] = {
    "sel",
    "discard",
    "disconnect",
    "bypassed",
    "sw_configs",
    "pe_enable",
    "instruction_mem",
    "per_fu_sw_configs",
    "visual_layout",
    "coordinates_semantic"};

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "fabric_artifact_invalid: " + message);
}

llvm::Expected<ArrayAttr> canonicalOperationList(::fabric::OpOp operation) {
  struct SchemaEntry {
    std::vector<std::uint8_t> identity;
    ::dataflow::OperationSchemaId schema;
  };
  llvm::SmallVector<SchemaEntry> schemas;
  schemas.reserve(operation.getOpList().size());
  for (Attribute attribute : operation.getOpList()) {
    auto symbol = dyn_cast<FlatSymbolRefAttr>(attribute);
    if (!symbol)
      return invalid("fabric.op op_list contains a non-symbol member");
    std::optional<::dataflow::OperationSchemaId> schema =
        ::dataflow::findOperationSchema(symbol.getValue());
    if (!schema)
      return invalid("fabric.op op_list contains an unregistered schema");
    auto identity = ::dataflow::encodeOperationSchemaId(*schema);
    if (!identity)
      return identity.takeError();
    schemas.push_back({identity->bytes().vec(), *schema});
  }
  llvm::sort(schemas, [](const SchemaEntry &left, const SchemaEntry &right) {
    return left.identity < right.identity;
  });
  if (std::adjacent_find(schemas.begin(), schemas.end(),
                         [](const SchemaEntry &left, const SchemaEntry &right) {
                           return left.identity == right.identity;
                         }) != schemas.end())
    return invalid("fabric.op op_list contains a duplicate schema");
  llvm::SmallVector<Attribute> canonical;
  canonical.reserve(schemas.size());
  for (const SchemaEntry &entry : schemas)
    canonical.push_back(FlatSymbolRefAttr::get(
        operation.getContext(),
        ::dataflow::operationSchemaSpelling(entry.schema)));
  return ArrayAttr::get(operation.getContext(), canonical);
}

bool isCanonicalSupplementalAttribute(NamedAttribute attribute) {
  llvm::StringRef name = attribute.getName().getValue();
  if (name == ::fabric::kEntityIdAttrName ||
      name == ::fabric::kFuTemplateIdAttrName ||
      name == ::fabric::kMemoryEngineTemplateIdAttrName)
    return isa<::fabric::EntityIdAttr>(attribute.getValue());
  return name == ::fabric::kResourceContractRecordAttrName &&
         isa<DenseI8ArrayAttr>(attribute.getValue());
}

llvm::Expected<bool> hasDeclaredYieldRelaxation(::fabric::YieldOp yield) {
  ArrayAttr declared = yield.getDeclaredTypesAttr();
  if (!declared)
    return false;
  if (declared.size() != yield.getValues().size())
    return invalid("fabric.yield declared_types count is inconsistent");
  bool hasRelaxation = false;
  for (auto [ordinal, attribute] : llvm::enumerate(declared)) {
    auto type = dyn_cast<TypeAttr>(attribute);
    if (!type)
      return invalid("fabric.yield declared_types contains a non-type member");
    hasRelaxation |= type.getValue() != yield.getValues()[ordinal].getType();
  }
  return hasRelaxation;
}

} // namespace

llvm::Error stripFabricModuleAuthoringState(::fabric::ModuleOp root) {
  llvm::Error result = llvm::Error::success();
  root->walk([&](Operation *operation) {
    if (result)
      return WalkResult::interrupt();
    operation->removeAttr(::fabric::kEntityIdAttrName);
    operation->removeAttr(::fabric::kFuTemplateIdAttrName);
    operation->removeAttr(::fabric::kMemoryEngineTemplateIdAttrName);
    operation->removeAttr("domain_slots");
    operation->removeAttr("domain_assignments");
    if (!isa<::fabric::OpOp>(operation))
      operation->removeAttr(::fabric::kResourceContractRecordAttrName);

    if (auto fabricOperation = dyn_cast<::fabric::OpOp>(operation)) {
      auto canonical = canonicalOperationList(fabricOperation);
      if (!canonical) {
        result = canonical.takeError();
        return WalkResult::interrupt();
      }
      fabricOperation.setOpListAttr(*canonical);
    }
    if (auto yield = dyn_cast<::fabric::YieldOp>(operation)) {
      auto hasRelaxation = hasDeclaredYieldRelaxation(yield);
      if (!hasRelaxation) {
        result = hasRelaxation.takeError();
        return WalkResult::interrupt();
      }
      if (!*hasRelaxation)
        yield->removeAttr("declared_types");
    }

    if (auto semantic =
            operation->getAttrOfType<BoolAttr>("coordinates_semantic");
        semantic && semantic.getValue()) {
      result = invalid("authoring coordinates claim semantic authority");
      return WalkResult::interrupt();
    }
    for (llvm::StringLiteral name : moduleAuthoringOnlyAttrs)
      operation->removeAttr(name);
    return WalkResult::advance();
  });
  return result;
}

llvm::Error eraseElaboratedFabricModuleDeclarations(::fabric::ModuleOp root) {
  llvm::SmallVector<Operation *> declarations;
  root->walk<WalkOrder::PostOrder>([&](Operation *operation) {
    if (operation == root.getOperation())
      return;
    auto symbol = dyn_cast<SymbolOpInterface>(operation);
    if (symbol && symbol.getNameAttr())
      declarations.push_back(operation);
  });
  for (Operation *declaration : declarations) {
    for (Value result : declaration->getResults())
      if (!result.use_empty())
        return invalid("an elaborated declaration still has an SSA use");
    declaration->erase();
  }
  bool residualInstance = false;
  root->walk([&](::fabric::InstantiateOp) { residualInstance = true; });
  if (residualInstance)
    return invalid("a fully elaborated Fabric contains fabric.instantiate");
  return llvm::Error::success();
}

llvm::Error validateCanonicalFabricModulePayload(::fabric::ModuleOp root) {
  enum class Violation {
    None,
    AuthoringState,
    NamedDeclaration,
    CanonicalRelationCarrier,
    OperationListOrder,
    RedundantYieldDeclaration,
  };
  Violation violation = Violation::None;
  llvm::Error validationError = llvm::Error::success();
  std::string unregisteredAttribute;
  root->walk([&](Operation *operation) {
    if (auto fabricOperation = dyn_cast<::fabric::OpOp>(operation)) {
      auto canonical = canonicalOperationList(fabricOperation);
      if (!canonical) {
        validationError = canonical.takeError();
        return WalkResult::interrupt();
      }
      if (*canonical != fabricOperation.getOpListAttr()) {
        violation = Violation::OperationListOrder;
        return WalkResult::interrupt();
      }
    }
    if (auto yield = dyn_cast<::fabric::YieldOp>(operation)) {
      auto hasRelaxation = hasDeclaredYieldRelaxation(yield);
      if (!hasRelaxation) {
        validationError = hasRelaxation.takeError();
        return WalkResult::interrupt();
      }
      if (yield.getDeclaredTypesAttr() && !*hasRelaxation) {
        violation = Violation::RedundantYieldDeclaration;
        return WalkResult::interrupt();
      }
    }
    const bool misplacedModuleRelation =
        operation != root.getOperation() &&
        (operation->hasAttr("domain_slots") ||
         operation->hasAttr("domain_assignments"));
    const bool misplacedFuRelation =
        !isa<::fabric::FuOp>(operation) &&
        operation->hasAttr("capability_templates");
    if (misplacedModuleRelation || misplacedFuRelation) {
      violation = Violation::CanonicalRelationCarrier;
      return WalkResult::interrupt();
    }
    for (llvm::StringLiteral name : moduleAuthoringOnlyAttrs)
      if (operation->hasAttr(name)) {
        violation = Violation::AuthoringState;
        return WalkResult::interrupt();
      }
    if (operation != root.getOperation())
      if (auto symbol = dyn_cast<SymbolOpInterface>(operation))
        if (symbol.getNameAttr()) {
          violation = Violation::NamedDeclaration;
          return WalkResult::interrupt();
        }
    for (NamedAttribute attribute : operation->getDiscardableAttrDictionary()) {
      llvm::StringRef name = attribute.getName().getValue();
      if (isCanonicalSupplementalAttribute(attribute))
        continue;
      unregisteredAttribute = name.str();
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (validationError)
    return validationError;
  if (!unregisteredAttribute.empty())
    return invalid("canonical Module payload has an unregistered "
                   "discardable attribute '" +
                   unregisteredAttribute + "'");
  if (violation == Violation::AuthoringState)
    return invalid("canonical Module payload retains authoring-only state");
  if (violation == Violation::NamedDeclaration)
    return invalid("canonical Module payload retains a named declaration");
  if (violation == Violation::CanonicalRelationCarrier)
    return invalid("canonical relation is attached to a non-carrier");
  if (violation == Violation::OperationListOrder)
    return invalid("fabric.op op_list is not in canonical schema-ID order");
  if (violation == Violation::RedundantYieldDeclaration)
    return invalid("fabric.yield has redundant declared_types");
  return llvm::Error::success();
}

} // namespace loom::fabric::detail
