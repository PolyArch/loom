//===-- loom.cpp - Loom compiler driver -------------------------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Loom compiler entry point. The driver integrates
// clang frontend invocation with the Loom MLIR pipeline. It compiles C/C++
// sources to LLVM IR, imports to MLIR LLVM dialect, applies annotation
// extraction (global annotations, loop markers, intrinsic annotations), runs
// the LLVM-to-SCF conversion, SCF post-processing, and SCF-to-Handshake
// conversion to produce hardware-focused dataflow IR.
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/CodeGenOptions.h"
#include "clang/Basic/Version.h"
#include "clang/CodeGen/CodeGenAction.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/Job.h"
#include "clang/Driver/Tool.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/FrontendTool/Utils.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/VirtualFileSystem.h"

#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVMIR/Import.h"
#include "mlir/Conversion/ControlFlowToSCF/ControlFlowToSCF.h"
#include "mlir/Transforms/Passes.h"

#include "mlir/Parser/Parser.h"

#include "loom/Conversion/LLVMToSCF.h"
#include "loom/Conversion/SCFToHandshake.h"
#include "loom/Conversion/SCFPostProcess.h"
#include "loom/Dialect/Dataflow/DataflowDialect.h"
#include "loom/Dialect/Fabric/FabricDialect.h"

#include "circt/Dialect/ESI/ESIDialect.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/Handshake/HandshakeDialect.h"
#include "circt/Dialect/Seq/SeqDialect.h"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {

using AnnotationMap =
    llvm::StringMap<llvm::SmallVector<std::string, 4>>;

void AppendAnnotation(AnnotationMap &map, llvm::StringRef symbol,
                      llvm::StringRef annotation) {
  if (symbol.empty() || annotation.empty())
    return;
  auto &bucket = map[symbol];
  bucket.push_back(annotation.str());
}

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

struct ParsedArgs {
  std::vector<std::string> inputs;
  std::vector<std::string> driver_args;
  std::string output_path;
  std::string adg_path;
  bool as_clang = false;
  bool show_help = false;
  bool show_version = false;
  bool had_error = false;
};

void PrintUsage(llvm::StringRef prog) {
  llvm::outs() << "Usage: " << prog
               << " [options] <sources...> -o <output.llvm.ll>\n";
  llvm::outs() << "       " << prog
               << " --adg <file.fabric.mlir>\n";
  llvm::outs() << "       " << prog
               << " --as-clang [clang-options...]\n";
  llvm::outs() << "\n";
  llvm::outs() << "Compile and link C++ sources into a single LLVM IR file "
               << "and emit LLVM dialect MLIR.\n";
  llvm::outs() << "The MLIR output path is derived from -o by replacing "
               << ".llvm.ll or .ll with .mlir.\n";
  llvm::outs() << "\n";
  llvm::outs() << "ADG validation mode (--adg):\n";
  llvm::outs() << "  Parse a fabric MLIR file, run semantic verification, "
               << "and exit 0 (valid) or 1 (errors).\n";
  llvm::outs() << "\n";
  llvm::outs() << "Forwarded compile options include: -I, -D, -U, -std, -O, -g,"
               << " -isystem, -include.\n";
  llvm::outs() << "Linker options (-l, -L, -Wl, -shared, -static) are ignored.\n";
}

void PrintVersion() {
  llvm::outs() << "loom based on " << clang::getClangFullVersion() << "\n";
}

bool IsLinkerFlag(llvm::StringRef arg) {
  if (arg == "-l" || arg == "-L" || arg == "-Xlinker")
    return true;
  if (arg.starts_with("-l") || arg.starts_with("-L"))
    return true;
  if (arg.starts_with("-Wl,"))
    return true;
  if (arg == "-shared" || arg == "-static")
    return true;
  return false;
}

bool LinkerFlagConsumesValue(llvm::StringRef arg) {
  return arg == "-l" || arg == "-L" || arg == "-Xlinker";
}

bool ClangFlagConsumesValue(llvm::StringRef arg) {
  return arg == "-I" || arg == "-isystem" || arg == "-iquote" ||
         arg == "-idirafter" || arg == "-D" || arg == "-U" ||
         arg == "-include" || arg == "-include-pch" ||
         arg == "-imacros" || arg == "-isysroot" || arg == "-sysroot" ||
         arg == "-stdlib" || arg == "-std" || arg == "-target" ||
         arg == "-gcc-toolchain" || arg == "-MF" || arg == "-MT" ||
         arg == "-MQ" || arg == "-fmodule-file" ||
         arg == "-fmodule-map-file" || arg == "-resource-dir" ||
         arg == "-Xclang" || arg == "-Xpreprocessor" || arg == "-Xassembler";
}

bool IsDashX(llvm::StringRef arg) {
  return arg == "-x" || (arg.starts_with("-x") && arg.size() > 2);
}

bool HasResourceDirArg(const std::vector<std::string> &args) {
  for (size_t i = 0; i < args.size(); ++i) {
    llvm::StringRef arg(args[i]);
    if (arg == "-resource-dir")
      return true;
    if (arg.starts_with("-resource-dir="))
      return true;
  }
  return false;
}

ParsedArgs ParseArgs(int argc, char **argv) {
  ParsedArgs parsed;
  bool passthrough_inputs = false;

  for (int i = 1; i < argc; ++i) {
    llvm::StringRef arg(argv[i]);

    if (!passthrough_inputs && arg == "--") {
      passthrough_inputs = true;
      continue;
    }

    if (!passthrough_inputs) {
      if (arg == "-h" || arg == "--help") {
        parsed.show_help = true;
        continue;
      }
      if (arg == "--version") {
        parsed.show_version = true;
        continue;
      }
      if (arg == "--adg") {
        if (i + 1 >= argc) {
          llvm::errs() << "error: --adg requires a path\n";
          parsed.had_error = true;
          break;
        }
        parsed.adg_path = argv[++i];
        continue;
      }
      if (arg == "--as-clang") {
        parsed.as_clang = true;
        // Collect all remaining arguments as passthrough to clang.
        for (++i; i < argc; ++i)
          parsed.driver_args.emplace_back(argv[i]);
        break;
      }
      if (arg == "-o") {
        if (i + 1 >= argc) {
          llvm::errs() << "error: -o requires a path\n";
          parsed.had_error = true;
          break;
        }
        parsed.output_path = argv[++i];
        continue;
      }
      if (arg.starts_with("-o") && arg.size() > 2) {
        parsed.output_path = arg.substr(2).str();
        continue;
      }

      if (IsLinkerFlag(arg)) {
        if (LinkerFlagConsumesValue(arg) && i + 1 < argc)
          ++i;
        continue;
      }
    }

    if (!passthrough_inputs && !arg.empty() && arg[0] == '-') {
      parsed.driver_args.emplace_back(arg.str());
      if (ClangFlagConsumesValue(arg)) {
        if (i + 1 >= argc) {
          llvm::errs() << "error: option requires a value: " << arg << "\n";
          parsed.had_error = true;
          break;
        }
        parsed.driver_args.emplace_back(argv[++i]);
      }
      continue;
    }

    parsed.inputs.emplace_back(arg.str());
  }

  return parsed;
}

std::string DefaultOutputPath(const std::vector<std::string> &inputs) {
  if (inputs.empty())
    return "a.llvm.ll";

  if (inputs.size() == 1) {
    llvm::SmallString<256> path(inputs.front());
    llvm::StringRef stem = llvm::sys::path::stem(path);
    if (!stem.empty()) {
      llvm::SmallString<256> output;
      output.assign(stem);
      output.append(".llvm.ll");
      return std::string(output);
    }
  }

  return "a.llvm.ll";
}

std::vector<std::string> BuildDriverArgs(
    const std::vector<std::string> &user_args) {
  std::vector<std::string> args = user_args;

  bool has_opt_level = false;
  bool has_dash_x = false;
  bool has_compile_only = false;
  for (const auto &arg : args) {
    if (!arg.empty() && arg[0] == '-' && arg.size() >= 2 && arg[1] == 'O') {
      has_opt_level = true;
    }
    if (IsDashX(arg)) {
      has_dash_x = true;
    }
    if (arg == "-c" || arg == "-S" || arg == "-E") {
      has_compile_only = true;
    }
  }

  if (!has_dash_x) {
    args.emplace_back("-x");
    args.emplace_back("c++");
  }

  if (!has_compile_only) {
    args.emplace_back("-c");
  }

  if (!has_opt_level) {
    args.emplace_back("-O1");
  }
  args.emplace_back("-emit-llvm");
  args.emplace_back("-g");
  args.emplace_back("-gno-column-info");
  args.emplace_back("-fno-discard-value-names");

  return args;
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

} // namespace

int main(int argc, char **argv) {
  llvm::InitLLVM init_llvm(argc, argv);
  llvm::sys::PrintStackTraceOnErrorSignal(argv[0]);

  // Handle -cc1 re-invocation from --as-clang mode. The clang driver
  // spawns the same executable with -cc1 args for the compile step.
  if (argc > 1 && llvm::StringRef(argv[1]) == "-cc1") {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    llvm::InitializeNativeTargetAsmParser();

    clang::DiagnosticOptions cc1_diag_opts;
    auto cc1_diags = clang::CompilerInstance::createDiagnostics(
        *llvm::vfs::getRealFileSystem(), cc1_diag_opts);
    auto cc1_invocation = std::make_shared<clang::CompilerInvocation>();
    bool ok = clang::CompilerInvocation::CreateFromArgs(
        *cc1_invocation,
        llvm::ArrayRef(argv + 2, argv + argc),
        *cc1_diags);
    if (!ok)
      return 1;
    clang::CompilerInstance cc1_compiler(std::move(cc1_invocation));
    cc1_compiler.createDiagnostics();
    return clang::ExecuteCompilerInvocation(&cc1_compiler) ? 0 : 1;
  }

  ParsedArgs parsed = ParseArgs(argc, argv);
  if (parsed.show_help) {
    PrintUsage(argv[0]);
    return parsed.had_error ? 1 : 0;
  }
  if (parsed.show_version) {
    PrintVersion();
    return parsed.had_error ? 1 : 0;
  }
  if (parsed.had_error)
    return 1;

  // --as-clang mode: invoke clang driver with ADG include/link flags.
  if (parsed.as_clang) {
    std::string exe_path_clang =
        llvm::sys::fs::getMainExecutable(argv[0],
                                         reinterpret_cast<void *>(&main));

    clang::DiagnosticOptions as_clang_diag_opts;
    auto as_clang_diag_client =
        std::make_unique<clang::TextDiagnosticPrinter>(
            llvm::errs(), as_clang_diag_opts);
    auto as_clang_diags = clang::CompilerInstance::createDiagnostics(
        *llvm::vfs::getRealFileSystem(), as_clang_diag_opts,
        as_clang_diag_client.get(), /*ShouldOwnClient=*/false);

    clang::driver::Driver as_clang_driver(
        exe_path_clang, llvm::sys::getDefaultTargetTriple(),
        *as_clang_diags);
    as_clang_driver.setTitle("loom --as-clang");
    as_clang_driver.setCheckInputsExist(true);

    // Point to the loom resource directory (build/lib/clang) so the
    // driver emits the correct -resource-dir for -cc1 re-invocations.
    llvm::SmallString<256> resource_dir(
        llvm::sys::path::parent_path(   // bin/
            llvm::sys::path::parent_path(exe_path_clang)));  // build/
    llvm::sys::path::append(resource_dir, "lib", "clang");
    as_clang_driver.ResourceDir = std::string(resource_dir);

    std::vector<const char *> as_clang_args;
    as_clang_args.push_back(exe_path_clang.c_str());

    // Inject ADG include path.
    std::string include_flag =
        std::string("-I") + LOOM_ADG_INCLUDE_DIR;
    as_clang_args.push_back(include_flag.c_str());

    // Inject library search path and RPATH for libloom-sdk.so.
    std::string lib_path_flag =
        std::string("-L") + LOOM_ADG_LIB_DIR;
    std::string rpath_flag =
        std::string("-Wl,-rpath,") + LOOM_ADG_LIB_DIR;
    as_clang_args.push_back(lib_path_flag.c_str());
    as_clang_args.push_back(rpath_flag.c_str());

    // Link against loom-sdk shared library (bundles LoomADG + deps).
    std::vector<std::string> link_libs = {
        "-lloom-sdk",
        "-lstdc++", "-lm"
    };

    // Forward user arguments.
    for (const auto &arg : parsed.driver_args)
      as_clang_args.push_back(arg.c_str());

    // Append link libraries after user args.
    for (const auto &lib : link_libs)
      as_clang_args.push_back(lib.c_str());

    std::unique_ptr<clang::driver::Compilation> as_clang_compilation(
        as_clang_driver.BuildCompilation(as_clang_args));
    if (!as_clang_compilation)
      return 1;

    llvm::SmallVector<std::pair<int, const clang::driver::Command *>, 4>
        failing_commands;
    int as_clang_result = as_clang_driver.ExecuteCompilation(
        *as_clang_compilation, failing_commands);
    return as_clang_result;
  }

  // ADG validation mode: parse fabric MLIR and run verification.
  if (!parsed.adg_path.empty()) {
    mlir::MLIRContext adg_context;
    mlir::DialectRegistry adg_registry;
    adg_context.appendDialectRegistry(adg_registry);
    adg_context.getDiagEngine().registerHandler(
        [](mlir::Diagnostic &diag) {
          diag.print(llvm::errs());
          llvm::errs() << "\n";
          return mlir::success();
        });
    adg_context.getOrLoadDialect<mlir::arith::ArithDialect>();
    adg_context.getOrLoadDialect<mlir::math::MathDialect>();
    adg_context.getOrLoadDialect<mlir::memref::MemRefDialect>();
    adg_context.getOrLoadDialect<mlir::func::FuncDialect>();
    adg_context.getOrLoadDialect<loom::dataflow::DataflowDialect>();
    adg_context.getOrLoadDialect<loom::fabric::FabricDialect>();
    adg_context.getOrLoadDialect<circt::handshake::HandshakeDialect>();

    mlir::ParserConfig adg_parser_config(&adg_context);
    auto adg_module = mlir::parseSourceFile<mlir::ModuleOp>(
        parsed.adg_path, adg_parser_config);
    if (!adg_module)
      return 1;
    if (failed(mlir::verify(*adg_module)))
      return 1;
    return 0;
  }

  if (parsed.inputs.empty()) {
    llvm::errs() << "error: no input files\n";
    PrintUsage(argv[0]);
    return 1;
  }

  if (parsed.output_path.empty())
    parsed.output_path = DefaultOutputPath(parsed.inputs);

  if (!EnsureOutputDirectory(parsed.output_path))
    return 1;

  std::string exe_path =
      llvm::sys::fs::getMainExecutable(argv[0],
                                       reinterpret_cast<void *>(&main));

  std::vector<std::string> driver_args = BuildDriverArgs(parsed.driver_args);

  if (!HasResourceDirArg(driver_args)) {
    llvm::SmallString<256> resource_dir =
        llvm::sys::path::parent_path(exe_path);
    resource_dir = llvm::sys::path::parent_path(resource_dir);
    llvm::sys::path::append(resource_dir, "lib", "clang");
    driver_args.push_back("-resource-dir=" + resource_dir.str().str());
  }

  clang::DiagnosticOptions diag_opts;
  auto diag_client = std::make_unique<clang::TextDiagnosticPrinter>(
      llvm::errs(), diag_opts);
  auto diags = clang::CompilerInstance::createDiagnostics(
      *llvm::vfs::getRealFileSystem(), diag_opts, diag_client.get(),
      /*ShouldOwnClient=*/false);

  clang::driver::Driver driver(exe_path, llvm::sys::getDefaultTargetTriple(),
                               *diags);
  driver.setTitle("loom");
  driver.setCheckInputsExist(true);

  std::vector<const char *> command_line;
  command_line.reserve(1 + driver_args.size() + parsed.inputs.size());
  command_line.push_back(exe_path.c_str());
  for (const auto &arg : driver_args)
    command_line.push_back(arg.c_str());
  for (const auto &input : parsed.inputs)
    command_line.push_back(input.c_str());

  std::unique_ptr<clang::driver::Compilation> compilation(
      driver.BuildCompilation(command_line));
  if (!compilation) {
    llvm::errs() << "error: failed to build clang driver compilation\n";
    return 1;
  }

  llvm::LLVMContext context;
  std::unique_ptr<llvm::Module> linked_module;
  unsigned compiled_inputs = 0;

  for (const auto &job : compilation->getJobs()) {
    const auto &cmd = job;
    if (!IsCC1Command(cmd))
      continue;

    auto invocation = std::make_shared<clang::CompilerInvocation>();
    if (!clang::CompilerInvocation::CreateFromArgs(
            *invocation, cmd.getArguments(), *diags, argv[0])) {
      llvm::errs() << "error: failed to build compiler invocation\n";
      return 1;
    }

    if (invocation->getFrontendOpts().Inputs.empty()) {
      llvm::errs() << "error: missing input file in compiler invocation\n";
      return 1;
    }

    invocation->getCodeGenOpts().setDebugInfo(
        llvm::codegenoptions::FullDebugInfo);

    const std::string input =
        invocation->getFrontendOpts().Inputs.front().getFile().str();

    auto module = CompileInvocation(invocation, context);
    if (!module) {
      llvm::errs() << "error: failed to compile " << input << "\n";
      return 1;
    }

    if (!linked_module) {
      linked_module = std::move(module);
      compiled_inputs++;
      continue;
    }

    if (!module->getTargetTriple().empty() &&
        module->getTargetTriple() != linked_module->getTargetTriple()) {
      llvm::errs() << "error: target triple mismatch when linking " << input
                   << "\n";
      return 1;
    }

    if (!module->getDataLayout().isDefault() &&
        !linked_module->getDataLayout().isDefault() &&
        module->getDataLayout() != linked_module->getDataLayout()) {
      llvm::errs() << "error: data layout mismatch when linking " << input
                   << "\n";
      return 1;
    }

    llvm::Linker linker(*linked_module);
    if (linker.linkInModule(std::move(module))) {
      llvm::errs() << "error: failed to link " << input << "\n";
      return 1;
    }
    compiled_inputs++;
  }

  if (compiled_inputs == 0) {
    llvm::errs() << "error: no compilation jobs were generated\n";
    return 1;
  }

  if (compiled_inputs < parsed.inputs.size()) {
    llvm::errs() << "error: not all inputs were compiled\n";
    return 1;
  }

  std::string verify_errors;
  llvm::raw_string_ostream verify_stream(verify_errors);
  if (llvm::verifyModule(*linked_module, &verify_stream)) {
    llvm::errs() << "error: linked module verification failed\n";
    llvm::errs() << verify_stream.str() << "\n";
    return 1;
  }

  StripUnsupportedAttributes(*linked_module);

  AnnotationMap symbol_annotations = CollectGlobalAnnotations(*linked_module);

  std::error_code ec;
  llvm::raw_fd_ostream output(parsed.output_path, ec, llvm::sys::fs::OF_Text);
  if (ec) {
    llvm::errs() << "error: cannot write output file: "
                 << parsed.output_path << "\n";
    llvm::errs() << ec.message() << "\n";
    return 1;
  }

  linked_module->print(output, nullptr);
  output.flush();

  std::string mlir_output_path = DeriveMlirOutputPath(parsed.output_path);
  if (!EnsureOutputDirectory(mlir_output_path))
    return 1;

  mlir::MLIRContext mlir_context;
  mlir::DialectRegistry registry;
  mlir::func::registerInlinerExtension(registry);
  mlir_context.appendDialectRegistry(registry);
  mlir_context.getDiagEngine().registerHandler(
      [](mlir::Diagnostic &diag) {
        diag.print(llvm::errs());
        llvm::errs() << "\n";
        return mlir::success();
      });
  mlir_context.getOrLoadDialect<mlir::LLVM::LLVMDialect>();
  mlir_context.getOrLoadDialect<mlir::DLTIDialect>();
  mlir_context.getOrLoadDialect<mlir::arith::ArithDialect>();
  mlir_context.getOrLoadDialect<mlir::cf::ControlFlowDialect>();
  mlir_context.getOrLoadDialect<mlir::func::FuncDialect>();
  mlir_context.getOrLoadDialect<mlir::math::MathDialect>();
  mlir_context.getOrLoadDialect<mlir::memref::MemRefDialect>();
  mlir_context.getOrLoadDialect<mlir::scf::SCFDialect>();
  mlir_context.getOrLoadDialect<mlir::ub::UBDialect>();
  mlir_context.getOrLoadDialect<loom::dataflow::DataflowDialect>();
  mlir_context.getOrLoadDialect<loom::fabric::FabricDialect>();
  mlir_context.getOrLoadDialect<circt::handshake::HandshakeDialect>();
  mlir_context.getOrLoadDialect<circt::esi::ESIDialect>();
  mlir_context.getOrLoadDialect<circt::hw::HWDialect>();
  mlir_context.getOrLoadDialect<circt::seq::SeqDialect>();

  auto mlir_module = mlir::translateLLVMIRToModule(
      std::move(linked_module), &mlir_context,
      /*emitExpensiveWarnings=*/false,
      /*dropDICompositeTypeElements=*/false, /*loadAllDialects=*/false);
  if (!mlir_module) {
    llvm::errs() << "error: failed to translate LLVM IR to MLIR\n";
    return 1;
  }

  ApplySymbolAnnotations(*mlir_module, symbol_annotations);
  ApplyIntrinsicAnnotations(*mlir_module);
  ApplyLoopMarkerAnnotations(*mlir_module);

  if (failed(mlir::verify(*mlir_module))) {
    llvm::errs() << "error: MLIR verification failed\n";
    return 1;
  }

  std::error_code mlir_ec;
  llvm::raw_fd_ostream mlir_output(mlir_output_path, mlir_ec,
                                   llvm::sys::fs::OF_Text);
  if (mlir_ec) {
    llvm::errs() << "error: cannot write MLIR output file: "
                 << mlir_output_path << "\n";
    llvm::errs() << mlir_ec.message() << "\n";
    return 1;
  }

  mlir::OpPrintingFlags print_flags;
  print_flags.enableDebugInfo(true, false);
  mlir_module->print(mlir_output, print_flags);
  mlir_output.flush();

  std::string scf_output_path = DeriveScfOutputPath(parsed.output_path);
  if (!EnsureOutputDirectory(scf_output_path))
    return 1;

  mlir::PassManager pass_manager(&mlir_context);
  pass_manager.addPass(loom::createConvertLLVMToSCFPass());
  pass_manager.addPass(mlir::createCanonicalizerPass());
  pass_manager.addPass(mlir::createCSEPass());
  pass_manager.addPass(mlir::createMem2Reg());
  pass_manager.addPass(mlir::createCanonicalizerPass());
  pass_manager.addPass(mlir::createCSEPass());
  pass_manager.addPass(mlir::createLiftControlFlowToSCFPass());
  pass_manager.addPass(mlir::createLoopInvariantCodeMotionPass());
  pass_manager.addPass(mlir::createCanonicalizerPass());
  pass_manager.addPass(mlir::createCSEPass());
  pass_manager.addPass(loom::createUpliftWhileToForPass());
  pass_manager.addPass(mlir::createCanonicalizerPass());
  pass_manager.addPass(mlir::createCSEPass());
  pass_manager.addPass(loom::createAttachLoopAnnotationsPass());
  pass_manager.addPass(loom::createMarkWhileStreamablePass());
  if (failed(pass_manager.run(*mlir_module))) {
    llvm::errs() << "error: failed to convert to scf stage\n";
    return 1;
  }

  if (failed(mlir::verify(*mlir_module))) {
    llvm::errs() << "error: scf stage verification failed\n";
    return 1;
  }

  std::error_code scf_ec;
  llvm::raw_fd_ostream scf_output(scf_output_path, scf_ec,
                                  llvm::sys::fs::OF_Text);
  if (scf_ec) {
    llvm::errs() << "error: cannot write scf output file: "
                 << scf_output_path << "\n";
    llvm::errs() << scf_ec.message() << "\n";
    return 1;
  }

  mlir_module->print(scf_output, print_flags);
  scf_output.flush();

  std::string handshake_output_path =
      DeriveHandshakeOutputPath(parsed.output_path);
  if (!EnsureOutputDirectory(handshake_output_path))
    return 1;

  mlir::PassManager handshake_passes(&mlir_context);
  handshake_passes.addPass(loom::createSCFToHandshakeDataflowPass());
  handshake_passes.addPass(mlir::createCanonicalizerPass());
  handshake_passes.addPass(mlir::createCSEPass());

  if (failed(handshake_passes.run(*mlir_module))) {
    llvm::errs() << "error: failed to convert to handshake stage\n";
    return 1;
  }

  if (failed(mlir::verify(*mlir_module))) {
    llvm::errs() << "error: handshake stage verification failed\n";
    return 1;
  }

  std::error_code handshake_ec;
  llvm::raw_fd_ostream handshake_output(handshake_output_path, handshake_ec,
                                        llvm::sys::fs::OF_Text);
  if (handshake_ec) {
    llvm::errs() << "error: cannot write handshake output file: "
                 << handshake_output_path << "\n";
    llvm::errs() << handshake_ec.message() << "\n";
    return 1;
  }

  mlir_module->print(handshake_output, print_flags);
  handshake_output.flush();

  return 0;
}
