//===-- loom_pipeline.cpp - Compilation pipeline helpers ---------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include "loom_pipeline.h"

#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/SymbolTable.h"

#include "clang/CodeGen/CodeGenAction.h"
#include "clang/Driver/Tool.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "llvm/TargetParser/Host.h"

#include <optional>
#include <string>

namespace loom {
namespace pipeline {

namespace {

void AppendAnnotation(AnnotationMap &map, llvm::StringRef symbol,
                      llvm::StringRef annotation) {
  if (symbol.empty() || annotation.empty())
    return;
  auto &bucket = map[symbol];
  bucket.push_back(annotation.str());
}

std::optional<std::string> ExtractStringFromGlobal(
    mlir::LLVM::GlobalOp global) {
  if (!global)
    return std::nullopt;
  auto value_attr = global.getValueAttr();
  if (!value_attr)
    return std::nullopt;

  if (auto str_attr = mlir::dyn_cast<mlir::StringAttr>(value_attr)) {
    std::string value = str_attr.getValue().str();
    if (!value.empty() && value.back() == '\0')
      value.pop_back();
    return value;
  }

  auto dense_attr = mlir::dyn_cast<mlir::DenseElementsAttr>(value_attr);
  if (!dense_attr || !dense_attr.getElementType().isInteger(8))
    return std::nullopt;

  std::string value;
  value.reserve(dense_attr.getNumElements());
  for (const llvm::APInt &element : dense_attr.getValues<llvm::APInt>()) {
    char ch = static_cast<char>(element.getZExtValue());
    if (ch == '\0')
      break;
    value.push_back(ch);
  }
  return value;
}

mlir::Value StripPointerOps(mlir::Value value) {
  while (true) {
    if (auto bitcast = value.getDefiningOp<mlir::LLVM::BitcastOp>()) {
      value = bitcast.getArg();
      continue;
    }
    if (auto gep = value.getDefiningOp<mlir::LLVM::GEPOp>()) {
      value = gep.getBase();
      continue;
    }
    return value;
  }
}

std::optional<std::string> ExtractStringFromOperand(mlir::ModuleOp module,
                                                    mlir::Value value) {
  mlir::Value stripped = StripPointerOps(value);
  if (auto addr = stripped.getDefiningOp<mlir::LLVM::AddressOfOp>()) {
    auto global =
        module.lookupSymbol<mlir::LLVM::GlobalOp>(addr.getGlobalName());
    return ExtractStringFromGlobal(global);
  }
  return std::nullopt;
}

void AppendLoomAnnotation(mlir::Operation *op, llvm::StringRef annotation,
                          mlir::Builder &builder) {
  if (!op || annotation.empty())
    return;
  auto existing = op->getAttrOfType<mlir::ArrayAttr>("loom.annotations");
  llvm::SmallVector<mlir::Attribute, 4> values;
  if (existing)
    values.append(existing.begin(), existing.end());
  values.push_back(builder.getStringAttr(annotation));
  op->setAttr("loom.annotations", builder.getArrayAttr(values));
}

std::optional<int64_t> GetConstantInt(mlir::Value value) {
  auto const_op = value.getDefiningOp<mlir::LLVM::ConstantOp>();
  if (!const_op)
    return std::nullopt;
  auto int_attr = mlir::dyn_cast<mlir::IntegerAttr>(const_op.getValue());
  if (!int_attr)
    return std::nullopt;
  return int_attr.getInt();
}

std::string FormatLoopMarkerAnnotation(llvm::StringRef callee,
                                       mlir::LLVM::CallOp call) {
  llvm::SmallString<64> storage;
  llvm::raw_svector_ostream os(storage);
  auto args = call.getArgOperands();

  auto append_field = [&](llvm::StringRef label,
                          std::optional<int64_t> value) {
    os << " " << label << "=";
    if (value)
      os << *value;
    else
      os << "?";
  };

  if (callee == "__loom_loop_parallel") {
    os << "loom.loop.parallel";
    if (args.size() >= 1)
      append_field("degree", GetConstantInt(args[0]));
    if (args.size() >= 2)
      append_field("schedule", GetConstantInt(args[1]));
    return os.str().str();
  }
  if (callee == "__loom_loop_parallel_auto")
    return "loom.loop.parallel=auto";
  if (callee == "__loom_loop_no_parallel")
    return "loom.loop.no_parallel";
  if (callee == "__loom_loop_unroll") {
    os << "loom.loop.unroll";
    if (args.size() >= 1)
      append_field("factor", GetConstantInt(args[0]));
    return os.str().str();
  }
  if (callee == "__loom_loop_unroll_auto")
    return "loom.loop.unroll=auto";
  if (callee == "__loom_loop_no_unroll")
    return "loom.loop.no_unroll";
  if (callee == "__loom_loop_tripcount") {
    os << "loom.loop.tripcount";
    if (args.size() >= 1)
      append_field("typical", GetConstantInt(args[0]));
    if (args.size() >= 2)
      append_field("avg", GetConstantInt(args[1]));
    if (args.size() >= 3)
      append_field("min", GetConstantInt(args[2]));
    if (args.size() >= 4)
      append_field("max", GetConstantInt(args[3]));
    return os.str().str();
  }

  return "";
}

} // anonymous namespace

AnnotationMap CollectGlobalAnnotations(const llvm::Module &module) {
  AnnotationMap annotations;
  const auto *global = module.getNamedGlobal("llvm.global.annotations");
  if (!global || !global->hasInitializer())
    return annotations;

  const auto *array =
      llvm::dyn_cast<llvm::ConstantArray>(global->getInitializer());
  if (!array)
    return annotations;

  for (const llvm::Value *operand : array->operands()) {
    const auto *entry = llvm::dyn_cast<llvm::ConstantStruct>(operand);
    if (!entry || entry->getNumOperands() < 2)
      continue;

    const llvm::Value *annotated =
        entry->getOperand(0)->stripPointerCasts();
    const auto *annotated_gv = llvm::dyn_cast<llvm::GlobalValue>(annotated);
    if (!annotated_gv)
      continue;

    llvm::StringRef annotation;
    if (!llvm::getConstantStringInfo(entry->getOperand(1), annotation))
      continue;

    AppendAnnotation(annotations, annotated_gv->getName(), annotation);
  }

  return annotations;
}

std::string DeriveMlirOutputPath(llvm::StringRef output_path) {
  if (output_path.ends_with(".llvm.ll")) {
    llvm::StringRef base = output_path.drop_back(3);
    return (base + ".mlir").str();
  }
  if (output_path.ends_with(".ll")) {
    llvm::StringRef base = output_path.drop_back(3);
    return (base + ".mlir").str();
  }
  return (output_path + ".mlir").str();
}

std::string DeriveScfOutputPath(llvm::StringRef output_path) {
  if (output_path.ends_with(".llvm.ll")) {
    llvm::StringRef base =
        output_path.drop_back(sizeof(".llvm.ll") - 1);
    return (base + ".scf.mlir").str();
  }
  if (output_path.ends_with(".ll")) {
    llvm::StringRef base = output_path.drop_back(3);
    return (base + ".scf.mlir").str();
  }
  return (output_path + ".scf.mlir").str();
}

std::string DeriveHandshakeOutputPath(llvm::StringRef output_path) {
  if (output_path.ends_with(".llvm.ll")) {
    llvm::StringRef base =
        output_path.drop_back(sizeof(".llvm.ll") - 1);
    return (base + ".handshake.mlir").str();
  }
  if (output_path.ends_with(".ll")) {
    llvm::StringRef base = output_path.drop_back(3);
    return (base + ".handshake.mlir").str();
  }
  return (output_path + ".handshake.mlir").str();
}

void ApplySymbolAnnotations(mlir::ModuleOp module,
                            const AnnotationMap &annotations) {
  if (annotations.empty())
    return;
  mlir::Builder builder(module.getContext());

  module.walk([&](mlir::Operation *op) {
    auto name_attr = op->getAttrOfType<mlir::StringAttr>(
        mlir::SymbolTable::getSymbolAttrName());
    if (!name_attr)
      return;
    auto it = annotations.find(name_attr.getValue());
    if (it == annotations.end())
      return;
    for (const std::string &value : it->second)
      AppendLoomAnnotation(op, value, builder);
  });
}

void ApplyLoopMarkerAnnotations(mlir::ModuleOp module) {
  mlir::Builder builder(module.getContext());
  module.walk([&](mlir::LLVM::CallOp call) {
    auto callee = call.getCallee();
    if (!callee)
      return;
    llvm::StringRef callee_name = *callee;
    if (!callee_name.starts_with("__loom_loop_"))
      return;
    std::string annotation = FormatLoopMarkerAnnotation(callee_name, call);
    if (annotation.empty())
      return;
    AppendLoomAnnotation(call.getOperation(), annotation, builder);
  });
}

void ApplyIntrinsicAnnotations(mlir::ModuleOp module) {
  mlir::Builder builder(module.getContext());
  module.walk([&](mlir::LLVM::CallIntrinsicOp call) {
    llvm::StringRef intrin_name = call.getIntrin();
    if (intrin_name.empty())
      return;
    if (!intrin_name.starts_with("llvm.var.annotation") &&
        !intrin_name.starts_with("llvm.ptr.annotation") &&
        !intrin_name.starts_with("llvm.annotation"))
      return;

    auto args = call.getArgs();
    if (args.size() < 2)
      return;

    std::optional<std::string> annotation =
        ExtractStringFromOperand(module, args[1]);
    if (!annotation || annotation->empty())
      return;

    mlir::Operation *target = args[0].getDefiningOp();
    if (!target)
      target = call.getOperation();
    AppendLoomAnnotation(target, *annotation, builder);
  });
}

bool IsCC1Command(const clang::driver::Command &cmd) {
  if (cmd.getCreator().isLinkJob())
    return false;
  for (const auto *arg : cmd.getArguments()) {
    if (llvm::StringRef(arg) == "-cc1")
      return true;
  }
  return false;
}

std::unique_ptr<llvm::Module> CompileInvocation(
    const std::shared_ptr<clang::CompilerInvocation> &invocation,
    llvm::LLVMContext &context) {
  if (invocation->getTargetOpts().Triple.empty())
    invocation->getTargetOpts().Triple = llvm::sys::getDefaultTargetTriple();

  clang::CompilerInstance compiler(invocation);
  compiler.createDiagnostics();
  compiler.createFileManager();
  compiler.createSourceManager();

  auto action = std::make_unique<clang::EmitLLVMOnlyAction>(&context);
  if (!compiler.ExecuteAction(*action))
    return nullptr;

  return action->takeModule();
}

bool EnsureOutputDirectory(llvm::StringRef output_path) {
  llvm::SmallString<256> dir = llvm::sys::path::parent_path(output_path);
  if (dir.empty())
    return true;

  std::error_code ec = llvm::sys::fs::create_directories(dir);
  if (ec) {
    llvm::errs() << "error: cannot create output directory: " << dir << "\n";
    llvm::errs() << ec.message() << "\n";
    return false;
  }

  return true;
}

void StripUnsupportedAttributes(llvm::Module &module) {
  const llvm::StringRef attr_name = "nocreateundeforpoison";
  const auto attr_kind = llvm::Attribute::getAttrKindFromName(attr_name);

  for (llvm::Function &func : module) {
    if (attr_kind != llvm::Attribute::None && func.hasFnAttribute(attr_kind))
      func.removeFnAttr(attr_kind);
    if (func.hasFnAttribute(attr_name))
      func.removeFnAttr(attr_name);

    for (unsigned index = 0; index < func.arg_size(); ++index) {
      if (attr_kind != llvm::Attribute::None &&
          func.hasParamAttribute(index, attr_kind)) {
        func.removeParamAttr(index, attr_kind);
      }
      if (func.hasParamAttribute(index, attr_name))
        func.removeParamAttr(index, attr_name);
    }

    if (attr_kind != llvm::Attribute::None)
      func.removeRetAttr(attr_kind);
  }

  for (llvm::Function &func : module) {
    for (llvm::Instruction &inst : llvm::instructions(func)) {
      auto *call = llvm::dyn_cast<llvm::CallBase>(&inst);
      if (!call)
        continue;

      if (attr_kind != llvm::Attribute::None && call->hasFnAttr(attr_kind))
        call->removeFnAttr(attr_kind);
      if (call->hasFnAttr(attr_name))
        call->removeFnAttr(attr_name);

      for (unsigned index = 0; index < call->arg_size(); ++index) {
        if (attr_kind != llvm::Attribute::None) {
          if (call->getParamAttr(index, attr_kind).isValid())
            call->removeParamAttr(index, attr_kind);
        }
        if (call->getParamAttr(index, attr_name).isValid())
          call->removeParamAttr(index, attr_name);
      }

      if (attr_kind != llvm::Attribute::None)
        call->removeRetAttr(attr_kind);
    }
  }
}

} // namespace pipeline
} // namespace loom
