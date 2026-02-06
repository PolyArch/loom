//===-- FabricTopOps.cpp - Fabric top-level operation impls ------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include "loom/Dialect/Fabric/FabricOps.h"
#include "loom/Dialect/Dataflow/DataflowTypes.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace loom::fabric;

//===----------------------------------------------------------------------===//
// ModuleOp parse/print
//
// fabric.module @name(%arg0: T0, %arg1: T1) -> (R0, R1) { body }
//===----------------------------------------------------------------------===//

ParseResult ModuleOp::parse(OpAsmParser &parser, OperationState &result) {
  // Parse @sym_name.
  StringAttr symName;
  if (parser.parseSymbolName(symName))
    return failure();
  result.addAttribute(getSymNameAttrName(result.name), symName);

  // Parse argument list: (%arg0: T0, %arg1: T1, ...).
  SmallVector<OpAsmParser::Argument> entryArgs;
  if (parser.parseArgumentList(entryArgs, OpAsmParser::Delimiter::Paren,
                               /*allowType=*/true))
    return failure();

  SmallVector<Type> argTypes;
  for (auto &arg : entryArgs)
    argTypes.push_back(arg.type);

  // Parse -> (result_types).
  SmallVector<Type> resultTypes;
  if (parser.parseArrow() || parser.parseLParen())
    return failure();
  if (failed(parser.parseOptionalRParen())) {
    if (parser.parseTypeList(resultTypes) || parser.parseRParen())
      return failure();
  }

  auto fnType = FunctionType::get(parser.getContext(), argTypes, resultTypes);
  result.addAttribute(getFunctionTypeAttrName(result.name),
                      TypeAttr::get(fnType));

  // Parse region body.
  auto *body = result.addRegion();
  if (parser.parseRegion(*body, entryArgs))
    return failure();
  if (body->empty())
    body->emplaceBlock();

  return success();
}

void ModuleOp::print(OpAsmPrinter &p) {
  p << " @" << getSymName() << "(";

  Block &entryBlock = getBody().front();
  llvm::interleaveComma(entryBlock.getArguments(), p,
                        [&](BlockArgument arg) {
                          p.printRegionArgument(arg);
                        });
  p << ")";

  p << " -> (";
  llvm::interleaveComma(getFunctionType().getResults(), p,
                        [&](Type t) { p.printType(t); });
  p << ")";

  p << " ";
  p.printRegion(getBody(), /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/true);
}

//===----------------------------------------------------------------------===//
// ModuleOp verify
//===----------------------------------------------------------------------===//

LogicalResult ModuleOp::verify() {
  auto fnType = getFunctionType();

  // Port ordering: memref* -> native* -> tagged*.
  auto checkOrdering = [&](ArrayRef<Type> types,
                           StringRef label) -> LogicalResult {
    // 0 = memref, 1 = native, 2 = tagged
    unsigned lastCat = 0;
    for (Type t : types) {
      unsigned cat;
      if (isa<MemRefType>(t))
        cat = 0;
      else if (isa<dataflow::TaggedType>(t))
        cat = 2;
      else
        cat = 1;
      if (cat < lastCat)
        return emitOpError(label)
               << " must follow port ordering: memref*, native*, tagged*";
      lastCat = cat;
    }
    return success();
  };

  if (failed(checkOrdering(fnType.getInputs(), "inputs")))
    return failure();
  if (failed(checkOrdering(fnType.getResults(), "outputs")))
    return failure();

  // Body must have at least one non-terminator.
  Block &body = getBody().front();
  bool hasOp = false;
  for (auto &op : body) {
    if (!op.hasTrait<OpTrait::IsTerminator>()) {
      hasOp = true;
      break;
    }
  }
  if (!hasOp)
    return emitOpError(
        "body must contain at least one non-terminator operation");

  // Yield operand types must match result types.
  auto yield = cast<YieldOp>(body.getTerminator());
  if (yield.getOperands().size() != fnType.getNumResults())
    return emitOpError("yield operand count (")
           << yield.getOperands().size() << ") must match result count ("
           << fnType.getNumResults() << ")";

  for (auto [idx, pair] : llvm::enumerate(
           llvm::zip(yield.getOperandTypes(), fnType.getResults()))) {
    if (std::get<0>(pair) != std::get<1>(pair))
      return emitOpError("yield operand #")
             << idx << " type " << std::get<0>(pair)
             << " must match result type " << std::get<1>(pair);
  }

  return success();
}

//===----------------------------------------------------------------------===//
// InstanceOp verify
//===----------------------------------------------------------------------===//

LogicalResult InstanceOp::verify() {
  // Full symbol resolution is deferred to a separate validation pass.
  return success();
}
