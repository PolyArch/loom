//===-- LLVMToSCFGlobals.cpp - LLVM global conversion ----------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// This file implements conversion of LLVM globals into memref globals.
//
//===----------------------------------------------------------------------===//

#include "loom/Conversion/LLVMToSCF.h"

using namespace mlir;

namespace loom::llvm_to_scf {

LogicalResult convertGlobals(ModuleOp module, OpBuilder &builder,
                             llvm::StringMap<ConvertedGlobal> &out) {
  llvm::SmallVector<LLVM::GlobalOp, 8> globals;
  for (auto global : module.getOps<LLVM::GlobalOp>())
    globals.push_back(global);

  for (LLVM::GlobalOp global : globals) {
    if (global.getName() == "llvm.global.annotations") {
      global.erase();
      continue;
    }
    Attribute valueAttr = global.getValueAttr();
    if (!valueAttr)
      continue;

    std::string originalName = global.getName().str();
    std::string renamedName = originalName + ".llvm";
    global.setSymName(renamedName);

    SmallVector<int64_t, 4> dims;
    Type scalar = GetScalarType(global.getType(), dims);
    scalar = NormalizeScalarType(scalar, module.getContext());
    if (!llvm::isa<IntegerType>(scalar) &&
        !llvm::isa<FloatType>(scalar)) {
      global.emitError("unsupported global element type");
      return failure();
    }

    int64_t totalCount = 1;
    for (int64_t dim : dims)
      totalCount *= dim;

    MemRefType memrefType =
        MemRefType::get({totalCount}, scalar, MemRefLayoutAttrInterface(),
                        global.getAddrSpaceAttr());

    Attribute initAttr;
    if (auto strAttr = llvm::dyn_cast<StringAttr>(valueAttr)) {
      std::string data = strAttr.getValue().str();
      SmallVector<int8_t, 16> bytes;
      bytes.reserve(data.size());
      for (char ch : data)
        bytes.push_back(static_cast<int8_t>(ch));
      auto tensorType =
          RankedTensorType::get({static_cast<int64_t>(bytes.size())},
                                IntegerType::get(module.getContext(), 8));
      llvm::ArrayRef<int8_t> byteRef(bytes.data(), bytes.size());
      initAttr = DenseElementsAttr::get(tensorType, byteRef);
    } else if (auto denseAttr =
                   llvm::dyn_cast<DenseElementsAttr>(valueAttr)) {
      initAttr = denseAttr;
    } else if (auto intAttr = llvm::dyn_cast<IntegerAttr>(valueAttr)) {
      auto tensorType = RankedTensorType::get({1}, scalar);
      initAttr = DenseElementsAttr::get(tensorType, {intAttr.getValue()});
    } else if (auto floatAttr = llvm::dyn_cast<FloatAttr>(valueAttr)) {
      auto tensorType = RankedTensorType::get({1}, scalar);
      initAttr = DenseElementsAttr::get(tensorType, {floatAttr.getValue()});
    } else {
      global.emitError("unsupported global initializer");
      return failure();
    }

    builder.setInsertionPoint(global);
    auto newGlobal = memref::GlobalOp::create(builder,
        global.getLoc(), originalName, StringAttr(), memrefType, initAttr,
        global.getConstant(), global.getAlignmentAttr());
    CopyLoomAnnotations(global.getOperation(), newGlobal.getOperation());
    out[originalName] = {memrefType, global};
    (void)newGlobal;
  }
  return success();
}

} // namespace loom::llvm_to_scf
