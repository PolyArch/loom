//===-- LLVMToSCF.h - LLVM to SCF conversion pass ---------------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// This header declares the LLVM-to-SCF conversion pass, which converts LLVM
// dialect operations to SCF, arith, memref, and func dialects. This pass is
// a key component of the Loom MLIR pipeline for processing C/C++ input.
//
// It also provides internal helper utilities for the conversion, including
// type normalization, pointer tracking, intrinsic conversion, annotation
// handling, and the entry points for global and function conversion.
//
//===----------------------------------------------------------------------===//

#ifndef LOOM_CONVERSION_LLVMTOSCF_H
#define LOOM_CONVERSION_LLVMTOSCF_H

#include <memory>
#include <optional>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"

namespace loom {

std::unique_ptr<mlir::Pass> createConvertLLVMToSCFPass();

} // namespace loom

namespace loom::llvm_to_scf {

// --- Support utilities ---

struct PointerInfo {
  mlir::Value base;
  mlir::Value index;
  mlir::Type elementType;
};

enum class StdMinMaxKind { Minimum, Maximum };

enum class StdMinMaxScalarKind {
  SignedIntKind,
  UnsignedIntKind,
  FloatKind,
  UnknownKind
};

void CopyLoomAnnotations(mlir::Operation *src, mlir::Operation *dst);
void MergeLoomAnnotationList(llvm::SmallVectorImpl<mlir::StringAttr> &dst,
                             mlir::ArrayAttr src);
mlir::ArrayAttr BuildLoomAnnotationArray(mlir::MLIRContext *context,
                                         llvm::ArrayRef<mlir::StringAttr> attrs);

bool IsPointerType(mlir::Type type);
bool IsRawPointerCallee(llvm::StringRef callee);
mlir::Type GetScalarType(mlir::Type type, llvm::SmallVectorImpl<int64_t> &dims);
mlir::Type NormalizeScalarType(mlir::Type type, mlir::MLIRContext *context);
int64_t GetByteSize(mlir::Type type);

mlir::Value BuildIndexConstant(mlir::OpBuilder &builder, mlir::Location loc,
                               int64_t value);
mlir::Value ScaleIndexBetweenElementTypes(mlir::OpBuilder &builder,
                                          mlir::Location loc,
                                          mlir::Value index,
                                          mlir::Type fromType,
                                          mlir::Type toType);
std::optional<mlir::Value>
BuildMemsetFillValue(mlir::OpBuilder &builder, mlir::Location loc,
                     mlir::Value fillVal, mlir::Type elemType);
mlir::Value ToIndexValue(mlir::OpBuilder &builder, mlir::Location loc,
                         mlir::Value value);

using VectorMapT = llvm::DenseMap<mlir::Value, llvm::SmallVector<mlir::Value, 8>>;

std::optional<mlir::Value>
LookupValue(const llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
            mlir::Value key);
std::optional<PointerInfo>
LookupPointer(const llvm::DenseMap<mlir::Value, PointerInfo> &ptrMap,
              mlir::Value key);
llvm::SmallVector<mlir::Value, 8> *LookupVector(VectorMapT &vectorMap,
                                                 mlir::Value key);
llvm::SmallVector<mlir::Value, 8>
ScalarizeDenseConstant(mlir::OpBuilder &builder, mlir::Location loc,
                       mlir::DenseElementsAttr attr);

std::optional<mlir::Type> GuessPointerElementTypeFromValue(mlir::Value value);
bool IsI8Type(mlir::Type type);
std::optional<mlir::Type> InferPointerElementType(mlir::Value value);
std::optional<mlir::Type>
InferPointerElementTypeFromCallSites(mlir::LLVM::LLVMFuncOp func,
                                     unsigned argIndex);

mlir::MemRefType MakeMemRefType(mlir::Type elementType,
                                mlir::Attribute memorySpace = {});
mlir::MemRefType MakeStridedMemRefType(mlir::Type elementType,
                                       mlir::Attribute memorySpace = {});
mlir::Value StripIndexCasts(mlir::Value value);
std::optional<int64_t> GetConstantIntValue(mlir::Value value);
bool IsZeroIndex(mlir::Value value);
mlir::Value MaterializeSubview(mlir::OpBuilder &builder, mlir::Location loc,
                               mlir::Value base, mlir::Value offset,
                               mlir::Value length);
mlir::Value MaterializeMemrefPointer(mlir::OpBuilder &builder, mlir::Location loc,
                                     const PointerInfo &info);
mlir::Value MaterializeLLVMPointer(mlir::OpBuilder &builder, mlir::Location loc,
                                   const PointerInfo &info);

std::optional<mlir::Value> ConvertLLVMConstant(mlir::OpBuilder &builder,
                                               mlir::LLVM::ConstantOp op);
std::optional<mlir::Value>
ConvertMathCall(mlir::OpBuilder &builder, mlir::Location loc,
                llvm::StringRef callee, mlir::ValueRange operands);

bool IsStdMinMaxName(llvm::StringRef callee, StdMinMaxKind &kind);
StdMinMaxScalarKind ParseStdMinMaxScalarKind(llvm::StringRef callee);
std::optional<PointerInfo>
BuildPointerSelect(mlir::OpBuilder &builder, mlir::Location loc,
                   mlir::Value cond, const PointerInfo &lhs,
                   const PointerInfo &rhs, bool trueSelectsRhs);
std::optional<mlir::Value>
ConvertStdMinMaxScalarCall(mlir::OpBuilder &builder, mlir::Location loc,
                           llvm::StringRef callee,
                           mlir::ValueRange operands);

mlir::arith::CmpIPredicate
ConvertICmpPredicate(mlir::LLVM::ICmpPredicate pred);
mlir::arith::CmpFPredicate
ConvertFCmpPredicate(mlir::LLVM::FCmpPredicate pred);

// --- Global and function conversion entry points ---

struct ConvertedGlobal {
  mlir::MemRefType type;
  mlir::LLVM::GlobalOp oldGlobal;
};

mlir::LogicalResult
convertGlobals(mlir::ModuleOp module, mlir::OpBuilder &builder,
               llvm::StringMap<ConvertedGlobal> &out);

mlir::LogicalResult
convertFunction(mlir::ModuleOp module, mlir::LLVM::LLVMFuncOp func,
                mlir::OpBuilder &builder,
                llvm::StringMap<ConvertedGlobal> &globals,
                const llvm::StringSet<> &varargFunctions);

} // namespace loom::llvm_to_scf

#endif // LOOM_CONVERSION_LLVMTOSCF_H
