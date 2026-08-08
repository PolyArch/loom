#include "Fabric/IR/FabricOps.h"

#include "Fabric/IR/FabricDialect.h"
#include "Fabric/IR/FabricTypes.h"
#include "Fabric/IR/ModuleDomain.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

using namespace mlir;
using namespace fabric;

//===----------------------------------------------------------------------===//
// fabric.instantiate
//===----------------------------------------------------------------------===//
//
// Assembly form mirrors fabric.fu's `(%a : T [to T_inner], ...)` operand
// list, with a leading `@callee` symbol reference and a trailing arrow +
// result type list:
//
//   fabric.instantiate @callee
//       (%a : !fabric.bits<32> [to !fabric.bits<16>],
//        %m : memref<8xi32>)
//       -> (!fabric.bits<32>, memref<8xi32>)
//
// Width relaxation on the input direction is expressed via the optional
// `to <inner-type>` clause per operand (mirroring fabric.fu / fabric.pe /
// fabric.fifo / fabric.yield). The output direction is strict in this
// iteration: each result type must equal the target's declared output
// port type.

namespace {

template <typename OpTy>
static LogicalResult getNamedTargetPortTypes(OpTy op,
                                             SmallVectorImpl<Type> &inputs,
                                             SmallVectorImpl<Type> &outputs) {
  if (!op.getSymNameAttr())
    return failure();
  auto functionTypeAttr = op.getFunctionTypeAttr();
  if (!functionTypeAttr)
    return failure();
  auto functionType = dyn_cast<FunctionType>(functionTypeAttr.getValue());
  if (!functionType)
    return failure();
  inputs.append(functionType.getInputs().begin(),
                functionType.getInputs().end());
  outputs.append(functionType.getResults().begin(),
                 functionType.getResults().end());
  return success();
}

// Returns the declared input/output port types of a fabric symbol target.
// Named PE, switch, memory, and FU definitions carry their signature in a
// function_type attribute; anonymous forms are not legal instantiate targets.
static LogicalResult getTargetPortTypes(Operation *target,
                                        SmallVectorImpl<Type> &inputs,
                                        SmallVectorImpl<Type> &outputs) {
  if (auto m = dyn_cast<fabric::ModuleOp>(target)) {
    for (Type t : m.getFunctionType().getInputs())
      inputs.push_back(t);
    for (Type t : m.getFunctionType().getResults())
      outputs.push_back(t);
    return success();
  }
  if (auto pe = dyn_cast<PeOp>(target))
    return getNamedTargetPortTypes(pe, inputs, outputs);
  if (auto sw = dyn_cast<SwitchOp>(target))
    return getNamedTargetPortTypes(sw, inputs, outputs);
  if (auto mem = dyn_cast<MemOp>(target))
    return getNamedTargetPortTypes(mem, inputs, outputs);
  if (auto fu = dyn_cast<FuOp>(target))
    return getNamedTargetPortTypes(fu, inputs, outputs);
  return failure();
}

// Returns true if `target`'s op kind is a legal instantiate target given
// the parent op of the `fabric.instantiate` site. The parent is one of:
//   * `fabric::ModuleOp` body: module, PE, switch, or memory.
//   * `fabric::PeOp` body: `fabric.fu` only.
// Returns false otherwise; the caller emits a precise diagnostic.
static bool isLegalKindForParent(Operation *parent, Operation *target) {
  if (isa<mlir::ModuleOp>(parent))
    return false;
  if (isa<fabric::ModuleOp>(parent))
    return isa<fabric::ModuleOp, fabric::PeOp, fabric::SwitchOp, fabric::MemOp>(
        target);
  if (isa<fabric::PeOp>(parent))
    return isa<fabric::FuOp>(target);
  return false;
}

static LogicalResult decodeDomainBindings(
    InstantiateOp instantiate,
    SmallVectorImpl<ModuleInstanceDomainSlotBinding> &bindings) {
  auto decoded = decodeModuleInstanceDomainSlotBindings(
      instantiate.getDomainSlotBindingsAttr());
  if (!decoded)
    return instantiate.emitOpError("has malformed domain-slot bindings: ")
           << llvm::toString(decoded.takeError());
  bindings.append(decoded->begin(), decoded->end());
  return success();
}

static llvm::Expected<std::optional<ModuleDomainSlotCounts>>
decodeModuleSlotCounts(fabric::ModuleOp module) {
  if (!module.getDomainSlotsAttr())
    return std::nullopt;
  auto slots = decodeModuleDomainSlots(module.getDomainSlotsAttr());
  if (!slots)
    return slots.takeError();

  ModuleDomainSlotCounts counts;
  for (const loom::fabric::FabricModuleDomainSlotRef &slot : *slots) {
    loom::fabric::FabricOrdinal *count = nullptr;
    switch (slot.kind) {
    case loom::fabric::FabricClockResetKind::Clock:
      count = &counts.clocks;
      break;
    case loom::fabric::FabricClockResetKind::Reset:
      count = &counts.resets;
      break;
    }
    if (!count || slot.ordinal != *count)
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "Module domain slot inventory is not canonical and dense");
    ++*count;
  }
  return counts;
}

static LogicalResult validateModuleDomainBindings(
    InstantiateOp instantiate, fabric::ModuleOp child, fabric::ModuleOp parent,
    ArrayRef<ModuleInstanceDomainSlotBinding> bindings) {
  if (bindings.empty())
    return instantiate.emitOpError(
        "targeting a Module requires non-empty domain-slot bindings");

  ModuleDomainSlotCounts inferredChild;
  ModuleDomainSlotCounts requiredParent;
  for (const ModuleInstanceDomainSlotBinding &binding : bindings) {
    loom::fabric::FabricOrdinal *childCount = nullptr;
    loom::fabric::FabricOrdinal *parentCount = nullptr;
    switch (binding.kind) {
    case loom::fabric::FabricClockResetKind::Clock:
      childCount = &inferredChild.clocks;
      parentCount = &requiredParent.clocks;
      break;
    case loom::fabric::FabricClockResetKind::Reset:
      childCount = &inferredChild.resets;
      parentCount = &requiredParent.resets;
      break;
    }
    if (!childCount || !parentCount || binding.childSlotOrdinal != *childCount)
      return instantiate.emitOpError(
          "has invalid domain-slot bindings: bindings are not the canonical "
          "total child-slot relation");
    ++*childCount;
    if (binding.parentSlotOrdinal >= *parentCount)
      *parentCount = binding.parentSlotOrdinal + 1;
  }
  if (inferredChild.clocks == 0 || inferredChild.resets == 0)
    return instantiate.emitOpError(
        "has invalid domain-slot bindings: every Module relation requires "
        "Clock and Reset rows");

  auto childCounts = decodeModuleSlotCounts(child);
  if (!childCounts)
    return instantiate.emitOpError("cannot decode callee domain slots: ")
           << llvm::toString(childCounts.takeError());
  auto parentCounts = decodeModuleSlotCounts(parent);
  if (!parentCounts)
    return instantiate.emitOpError("cannot decode parent domain slots: ")
           << llvm::toString(parentCounts.takeError());

  if (llvm::Error error = validateModuleInstanceDomainSlotBindings(
          childCounts->value_or(inferredChild),
          parentCounts->value_or(requiredParent), bindings))
    return instantiate.emitOpError("has invalid domain-slot bindings: ")
           << llvm::toString(std::move(error));
  return success();
}

} // namespace

Operation *
fabric::resolveInstantiateTarget(InstantiateOp instantiate,
                                 SymbolTableCollection &symbolTables) {
  Operation *cursor = instantiate.getOperation();
  while (cursor) {
    Operation *symbolTable = SymbolTable::getNearestSymbolTable(cursor);
    if (!symbolTable)
      break;
    if (Operation *target = symbolTables.lookupSymbolIn(
            symbolTable, instantiate.getCalleeAttr()))
      return target;
    cursor = symbolTable->getParentOp();
  }
  return nullptr;
}

ParseResult InstantiateOp::parse(OpAsmParser &parser, OperationState &result) {
  // `@callee`
  FlatSymbolRefAttr callee;
  if (parser.parseAttribute(callee, "callee", result.attributes))
    return failure();

  // `(` operand-list `)`
  SmallVector<OpAsmParser::UnresolvedOperand, 4> operands;
  SmallVector<Type, 4> outerTypes;
  SmallVector<Type, 4> innerTypes;
  SMLoc operandsLoc = parser.getCurrentLocation();
  if (parser.parseLParen())
    return failure();
  if (failed(parser.parseOptionalRParen())) {
    auto parseOne = [&]() -> ParseResult {
      OpAsmParser::UnresolvedOperand op;
      Type outer;
      if (parser.parseOperand(op) || parser.parseColon() ||
          parser.parseType(outer))
        return failure();
      Type inner = outer;
      if (succeeded(parser.parseOptionalKeyword("to")))
        if (parser.parseType(inner))
          return failure();
      operands.push_back(op);
      outerTypes.push_back(outer);
      innerTypes.push_back(inner);
      return success();
    };
    if (parseOne())
      return failure();
    while (succeeded(parser.parseOptionalComma()))
      if (parseOne())
        return failure();
    if (parser.parseRParen())
      return failure();
  }

  // Resolve operand SSA types against the declared OUTER (source) types so
  // the IR's operand list matches the SSA producer side. Inner (target's
  // declared input) types are stashed as a property for the verifier and
  // the printer to recover.
  if (parser.resolveOperands(operands, outerTypes, operandsLoc,
                             result.operands))
    return failure();

  // Stash inner types when any of them differs from its outer; otherwise
  // leave the property empty so the no-relaxation case round-trips cleanly.
  bool anyDiffer = false;
  for (auto [o, i] : llvm::zip(outerTypes, innerTypes))
    if (o != i) {
      anyDiffer = true;
      break;
    }
  if (anyDiffer) {
    result.getOrAddProperties<Properties>().setInnerInputTypes(innerTypes);
  }

  // `-> ( T0, T1, ... )` or `-> T` or empty.
  SmallVector<Type, 4> resultTypes;
  if (parser.parseArrow())
    return failure();
  if (succeeded(parser.parseOptionalLParen())) {
    if (failed(parser.parseOptionalRParen())) {
      if (parser.parseTypeList(resultTypes) || parser.parseRParen())
        return failure();
    }
  } else {
    Type ty;
    if (parser.parseType(ty))
      return failure();
    resultTypes.push_back(ty);
  }
  result.addTypes(resultTypes);

  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  return success();
}

void InstantiateOp::print(OpAsmPrinter &p) {
  p << ' ';
  p.printAttributeWithoutType(getCalleeAttr());
  p << '(';
  // Recover inner (declared-target) types, defaulting to the operand's SSA
  // outer type when no relaxation was recorded.
  ArrayRef<Type> innerTypes = getInnerInputTypes();
  SmallVector<Type, 4> inner;
  inner.reserve(getInputs().size());
  if (!innerTypes.empty() && innerTypes.size() == getInputs().size()) {
    inner.append(innerTypes.begin(), innerTypes.end());
  } else {
    for (Value v : getInputs())
      inner.push_back(v.getType());
  }
  llvm::interleaveComma(llvm::zip(getInputs(), inner), p, [&](auto pair) {
    Value v;
    Type i;
    std::tie(v, i) = pair;
    p << v << " : " << v.getType();
    if (i && i != v.getType())
      p << " to " << i;
  });
  p << ')';
  p << " -> ";
  auto rTypes = getResultTypes();
  if (rTypes.size() == 1) {
    p << rTypes.front();
  } else {
    p << '(';
    llvm::interleaveComma(rTypes, p);
    p << ')';
  }
  // Elide attributes already serialized inline.
  SmallVector<StringRef, 1> elided{getCalleeAttrName().getValue()};
  p.printOptionalAttrDict(getOperation()->getAttrs(), elided);
}

LogicalResult InstantiateOp::verify() {
  if (failed(verifyInnerInputTypesProperty(getOperation(), getInputs(),
                                           getInnerInputTypes())))
    return failure();

  // Recover the inner (declared-target) input types if any. When absent,
  // the inner type equals the operand's outer SSA type for every port.
  ArrayRef<Type> innerTypes = getInnerInputTypes();
  SmallVector<Type, 4> inner;
  inner.reserve(getInputs().size());
  if (!innerTypes.empty()) {
    inner.append(innerTypes.begin(), innerTypes.end());
  } else {
    for (Value v : getInputs())
      inner.push_back(v.getType());
  }

  // Local sanity: per-operand outer/inner kind agreement and memref-exact.
  // (The cross-check against the callee's declared port type happens below
  // inside verifySymbolUses, where SymbolTable resolution is permitted.)
  for (auto [i, pair] : llvm::enumerate(llvm::zip(getInputs(), inner))) {
    Value v;
    Type innerTy;
    std::tie(v, innerTy) = pair;
    Type outerTy = v.getType();
    if (outerTy == innerTy)
      continue;
    if (!haveSameFabricModulePortKind(outerTy, innerTy))
      return emitOpError("operand #")
             << i << " outer type " << outerTy << " and declared inner type "
             << innerTy
             << " must share the same fabric kind (bits, bits_tag, memref); "
                "low-bit alignment / zero-fill applies on width relaxation";
    if (isa<MemRefType>(outerTy))
      return emitOpError("operand #")
             << i
             << ": memref operands cannot use the 'to <inner-type>' clause; "
                "memref types must match exactly";
  }

  SmallVector<ModuleInstanceDomainSlotBinding, 4> bindings;
  if (failed(decodeDomainBindings(*this, bindings)))
    return failure();
  return success();
}

LogicalResult
InstantiateOp::verifySymbolUses(::mlir::SymbolTableCollection &symbolTable) {
  // 1. Resolve the callee. Both fabric.module and fabric.pe carry the
  //    SymbolTable trait, so sibling-module lookup must walk OUT of the
  //    nearest SymbolTable scope when the symbol isn't found locally.
  //    Walk outward: starting from this op, then from each enclosing
  //    SymbolTable's parent, until a match is found or no SymbolTable
  //    remains. This is the explicit "reaches sibling top-level modules
  //    via SymbolTable lookup-up" semantics from the spec.
  Operation *target = resolveInstantiateTarget(*this, symbolTable);
  if (!target)
    return emitOpError("references undefined symbol '@") << getCallee() << "'";

  // 2. Confirm the parent-of-instantiate / target-kind table.
  Operation *parent = (*this)->getParentOp();
  if (!parent)
    return emitOpError(
        "'fabric.instantiate' must have an enclosing op (top-level "
        "builtin.module, fabric.module, or fabric.pe)");
  if (!isLegalKindForParent(parent, target)) {
    StringRef parentName = parent->getName().getStringRef();
    StringRef targetName = target->getName().getStringRef();
    if (isa<mlir::ModuleOp>(parent))
      return emitOpError("directly under builtin.module is not allowed");
    if (isa<fabric::ModuleOp>(parent))
      return emitOpError("inside a fabric.module body may only target "
                         "'fabric.module', 'fabric.pe', 'fabric.switch', or "
                         "'fabric.mem'; got target kind '")
             << targetName << "' for symbol '@" << getCallee() << "'";
    if (isa<fabric::PeOp>(parent))
      return emitOpError("inside a fabric.pe body may only target "
                         "'fabric.fu'; got target kind '")
             << targetName << "' for symbol '@" << getCallee() << "'";
    return emitOpError("has unsupported parent op '")
           << parentName << "' for fabric.instantiate";
  }

  SmallVector<ModuleInstanceDomainSlotBinding, 4> domainBindings;
  if (failed(decodeDomainBindings(*this, domainBindings)))
    return failure();
  if (!isa<fabric::ModuleOp>(target)) {
    if (!domainBindings.empty())
      return emitOpError(
          "a non-Module target cannot have domain-slot bindings");
  } else if (auto parentModule = dyn_cast<fabric::ModuleOp>(parent)) {
    if (failed(validateModuleDomainBindings(*this,
                                            cast<fabric::ModuleOp>(target),
                                            parentModule, domainBindings)))
      return failure();
  } else if (!domainBindings.empty()) {
    return emitOpError(
        "a Module target outside a fabric.module cannot bind domain slots");
  }

  // 3. Self-reference: the closest enclosing Symbol op of this
  //    fabric.instantiate must NOT be the target itself.
  Operation *enclosingSym = parent;
  while (enclosingSym &&
         !isa<fabric::ModuleOp, fabric::PeOp, fabric::FuOp>(enclosingSym))
    enclosingSym = enclosingSym->getParentOp();
  if (enclosingSym == target)
    return emitOpError("cannot instantiate the symbol that encloses it "
                       "(self-reference of '@")
           << getCallee() << "')";

  // 4. Forward-reference: the target op must precede this instantiate in
  //    the common ancestor block. Walk both ops up to a shared block, then
  //    use Operation::isBeforeInBlock.
  // Find the chain of parents for `target` and for `*this`. The lowest
  // common Block is where ordering is meaningful.
  auto opAncestorsInBlock = [](Operation *op, Block *blk) -> Operation * {
    while (op && op->getBlock() != blk)
      op = op->getParentOp();
    return op;
  };
  Operation *targetInOurChain = nullptr;
  Operation *ourSelfInOurChain = nullptr;
  // Walk this op's parent chain; for each ancestor block, see if `target`
  // (or one of its ancestors) is in the same block.
  Operation *cursor = getOperation();
  while (cursor) {
    Block *blk = cursor->getBlock();
    if (!blk)
      break;
    Operation *targetSibling = opAncestorsInBlock(target, blk);
    if (targetSibling) {
      targetInOurChain = targetSibling;
      ourSelfInOurChain = cursor;
      break;
    }
    cursor = cursor->getParentOp();
  }
  if (targetInOurChain && ourSelfInOurChain &&
      targetInOurChain->getBlock() == ourSelfInOurChain->getBlock()) {
    if (!targetInOurChain->isBeforeInBlock(ourSelfInOurChain))
      return emitOpError("forward reference to symbol '@")
             << getCallee()
             << "': the named definition must textually precede its "
                "fabric.instantiate use";
  }

  // 5./6. Operand and result port count checks.
  SmallVector<Type, 4> declaredIn, declaredOut;
  if (failed(getTargetPortTypes(target, declaredIn, declaredOut)))
    return emitOpError(
        "internal: target op kind unrecognized after legality check");
  if (declaredIn.size() != getInputs().size())
    return emitOpError("operand count (")
           << getInputs().size() << ") does not match callee '@" << getCallee()
           << "' input port count (" << declaredIn.size() << ')';
  if (declaredOut.size() != getResultTypes().size())
    return emitOpError("result count (")
           << getResultTypes().size() << ") does not match callee '@"
           << getCallee() << "' output port count (" << declaredOut.size()
           << ')';

  // Recover declared inner-input types stashed by the parser/builder.
  ArrayRef<Type> innerTypes = getInnerInputTypes();
  SmallVector<Type, 4> inner;
  inner.reserve(getInputs().size());
  if (!innerTypes.empty() && innerTypes.size() == getInputs().size()) {
    inner.append(innerTypes.begin(), innerTypes.end());
  } else {
    for (Value v : getInputs())
      inner.push_back(v.getType());
  }

  // 7. Per-input port: declared inner type (the operand's `to <T_inner>`
  //    clause, or operand SSA type when omitted) must equal the target's
  //    declared input port type. The instantiate-time width relaxation
  //    happens between the operand SSA outer type and the inner type, and
  //    is verified in InstantiateOp::verify(); here we only check the
  //    inner-vs-target alignment.
  for (auto [i, t] : llvm::enumerate(declaredIn)) {
    Type innerTy = inner[i];
    if (innerTy != t)
      return emitOpError("input #") << i << " declared inner type " << innerTy
                                    << " must equal callee '@" << getCallee()
                                    << "' input port type " << t;
  }
  // 8. Output direction is strict in this iteration: each result SSA
  //    type must equal the target's declared output port type.
  for (auto [i, pair] :
       llvm::enumerate(llvm::zip(getResultTypes(), declaredOut))) {
    Type r;
    Type t;
    std::tie(r, t) = pair;
    if (r != t)
      return emitOpError("result #")
             << i << " type " << r << " must equal callee '@" << getCallee()
             << "' output port type " << t
             << " (output direction is strict; no width relaxation)";
  }
  return success();
}
