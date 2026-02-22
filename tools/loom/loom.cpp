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
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
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
#include "loom/Dialect/Fabric/FabricOps.h"
#include "loom/Mapper/ADGFlattener.h"
#include "loom/Mapper/ConfigGen.h"
#include "loom/Mapper/DFGBuilder.h"
#include "loom/Mapper/Mapper.h"
#include "loom/Mapper/TechMapper.h"
#include "loom/Viz/DOTExporter.h"

#include "circt/Dialect/ESI/ESIDialect.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/Handshake/HandshakeDialect.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "circt/Dialect/Seq/SeqDialect.h"

#include "loom_args.h"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using loom::tool::ParsedArgs;
using loom::tool::ParseArgs;
using loom::tool::PrintUsage;
using loom::tool::PrintVersion;
using loom::tool::DefaultOutputPath;
using loom::tool::BuildDriverArgs;
using loom::tool::HasResourceDirArg;
using loom::tool::DeriveConfigBinPath;
using loom::tool::DeriveAddrHeaderPath;
using loom::tool::DeriveMappingJsonPath;

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

  // Incompatibility checks.
  if (!parsed.handshake_input.empty() && parsed.adg_path.empty()) {
    llvm::errs() << "error: --handshake-input requires --adg\n";
    return 1;
  }
  if (!parsed.handshake_input.empty() && !parsed.inputs.empty()) {
    llvm::errs() << "error: --handshake-input is incompatible with source files\n";
    return 1;
  }

  // Standalone visualization modes.
  if (!parsed.viz_dfg.empty() || !parsed.viz_adg.empty() ||
      !parsed.viz_mapped.empty()) {
    if (parsed.output_path.empty()) {
      llvm::errs() << "error: -o is required for visualization\n";
      return 1;
    }
    if (!EnsureOutputDirectory(parsed.output_path))
      return 1;

    // Set up shared MLIR context with all required dialects.
    mlir::MLIRContext viz_ctx;
    mlir::DialectRegistry viz_reg;
    mlir::func::registerInlinerExtension(viz_reg);
    viz_ctx.appendDialectRegistry(viz_reg);
    viz_ctx.getDiagEngine().registerHandler(
        [](mlir::Diagnostic &diag) {
          diag.print(llvm::errs());
          llvm::errs() << "\n";
          return mlir::success();
        });
    viz_ctx.getOrLoadDialect<mlir::arith::ArithDialect>();
    viz_ctx.getOrLoadDialect<mlir::math::MathDialect>();
    viz_ctx.getOrLoadDialect<mlir::memref::MemRefDialect>();
    viz_ctx.getOrLoadDialect<mlir::func::FuncDialect>();
    viz_ctx.getOrLoadDialect<mlir::scf::SCFDialect>();
    viz_ctx.getOrLoadDialect<loom::dataflow::DataflowDialect>();
    viz_ctx.getOrLoadDialect<loom::fabric::FabricDialect>();
    viz_ctx.getOrLoadDialect<circt::handshake::HandshakeDialect>();
    viz_ctx.getOrLoadDialect<circt::esi::ESIDialect>();
    viz_ctx.getOrLoadDialect<circt::hw::HWDialect>();
    viz_ctx.getOrLoadDialect<circt::seq::SeqDialect>();
    viz_ctx.getOrLoadDialect<mlir::LLVM::LLVMDialect>();
    viz_ctx.allowUnregisteredDialects();

    mlir::ParserConfig viz_parser_cfg(&viz_ctx);
    loom::viz::DOTOptions viz_opts;
    if (parsed.viz_mode == "detailed")
      viz_opts.mode = loom::viz::DOTMode::Detailed;

    if (!parsed.viz_dfg.empty()) {
      auto viz_mod = mlir::parseSourceFile<mlir::ModuleOp>(
          parsed.viz_dfg, viz_parser_cfg);
      if (!viz_mod) {
        llvm::errs() << "error: failed to parse: " << parsed.viz_dfg << "\n";
        return 1;
      }
      circt::handshake::FuncOp hs_func;
      viz_mod->walk([&](circt::handshake::FuncOp f) {
        if (!hs_func) hs_func = f;
      });
      if (!hs_func) {
        llvm::errs() << "error: no handshake.func in: " << parsed.viz_dfg << "\n";
        return 1;
      }
      loom::DFGBuilder dfg_builder;
      loom::Graph dfg = dfg_builder.build(hs_func);
      std::string dot = loom::viz::exportDFGDot(dfg, viz_opts);
      std::string html = loom::viz::generateHTML(dot, "DFG Visualization");

      std::error_code ec;
      llvm::raw_fd_ostream out(parsed.output_path, ec, llvm::sys::fs::OF_Text);
      if (ec) {
        llvm::errs() << "error: cannot write: " << parsed.output_path << "\n";
        return 1;
      }
      out << html;
      out.flush();
      return 0;
    }

    if (!parsed.viz_adg.empty()) {
      auto viz_mod = mlir::parseSourceFile<mlir::ModuleOp>(
          parsed.viz_adg, viz_parser_cfg);
      if (!viz_mod) {
        llvm::errs() << "error: failed to parse: " << parsed.viz_adg << "\n";
        return 1;
      }
      loom::fabric::ModuleOp fab_mod;
      viz_mod->walk([&](loom::fabric::ModuleOp m) {
        if (!fab_mod) fab_mod = m;
      });
      if (!fab_mod) {
        llvm::errs() << "error: no fabric.module in: " << parsed.viz_adg << "\n";
        return 1;
      }
      loom::ADGFlattener flattener;
      loom::Graph adg = flattener.flatten(fab_mod);
      std::string dot = loom::viz::exportADGDot(adg, viz_opts);
      std::string html = loom::viz::generateHTML(dot, "ADG Visualization");

      std::error_code ec;
      llvm::raw_fd_ostream out(parsed.output_path, ec, llvm::sys::fs::OF_Text);
      if (ec) {
        llvm::errs() << "error: cannot write: " << parsed.output_path << "\n";
        return 1;
      }
      out << html;
      out.flush();
      return 0;
    }

    if (!parsed.viz_mapped.empty()) {
      // Standalone mapped visualization: read mapping JSON, ADG, and DFG.
      if (parsed.adg_path.empty()) {
        llvm::errs() << "error: --viz-mapped requires --adg <fabric.mlir>\n";
        return 1;
      }
      if (parsed.handshake_input.empty()) {
        llvm::errs()
            << "error: --viz-mapped requires --handshake-input <handshake.mlir>\n";
        return 1;
      }

      // Read mapping JSON file.
      auto buf = llvm::MemoryBuffer::getFile(parsed.viz_mapped);
      if (!buf) {
        llvm::errs() << "error: cannot read: " << parsed.viz_mapped << "\n";
        return 1;
      }
      auto json_val = llvm::json::parse((*buf)->getBuffer());
      if (!json_val) {
        llvm::errs() << "error: invalid JSON: " << parsed.viz_mapped << "\n";
        return 1;
      }
      auto *root = json_val->getAsObject();
      if (!root) {
        llvm::errs() << "error: JSON root is not an object\n";
        return 1;
      }

      // Parse ADG fabric MLIR.
      auto adg_mod = mlir::parseSourceFile<mlir::ModuleOp>(
          parsed.adg_path, viz_parser_cfg);
      if (!adg_mod) {
        llvm::errs() << "error: failed to parse ADG: " << parsed.adg_path << "\n";
        return 1;
      }
      loom::fabric::ModuleOp fab_mod;
      adg_mod->walk([&](loom::fabric::ModuleOp m) { if (!fab_mod) fab_mod = m; });
      if (!fab_mod) {
        llvm::errs() << "error: no fabric.module in: " << parsed.adg_path << "\n";
        return 1;
      }

      // Parse Handshake MLIR.
      auto hs_mod = mlir::parseSourceFile<mlir::ModuleOp>(
          parsed.handshake_input, viz_parser_cfg);
      if (!hs_mod) {
        llvm::errs() << "error: failed to parse: " << parsed.handshake_input << "\n";
        return 1;
      }
      circt::handshake::FuncOp hs_func;
      hs_mod->walk([&](circt::handshake::FuncOp f) { if (!hs_func) hs_func = f; });
      if (!hs_func) {
        llvm::errs() << "error: no handshake.func in: "
                     << parsed.handshake_input << "\n";
        return 1;
      }

      // Build DFG and ADG graphs.
      loom::DFGBuilder dfg_builder;
      loom::Graph dfg = dfg_builder.build(hs_func);
      loom::ADGFlattener flattener;
      loom::Graph adg = flattener.flatten(fab_mod);

      // Reconstruct MappingState from JSON.
      loom::MappingState state;
      state.init(dfg, adg);
      if (auto *pl = root->getObject("placement")) {
        for (auto &kv : *pl) {
          unsigned sw_id;
          if (llvm::StringRef(kv.first).getAsInteger(10, sw_id))
            continue;
          auto *entry = kv.second.getAsObject();
          if (!entry)
            continue;
          auto hw_val = entry->getInteger("hwNode");
          if (!hw_val)
            continue;
          auto hw_id = static_cast<loom::IdIndex>(*hw_val);
          if (sw_id < state.swNodeToHwNode.size()) {
            state.swNodeToHwNode[sw_id] = hw_id;
            if (hw_id < state.hwNodeToSwNodes.size())
              state.hwNodeToSwNodes[hw_id].push_back(sw_id);
          }
        }
      }
      if (auto *pb = root->getObject("portBinding")) {
        for (auto &kv : *pb) {
          unsigned sp;
          if (llvm::StringRef(kv.first).getAsInteger(10, sp))
            continue;
          auto hp = kv.second.getAsInteger();
          if (!hp)
            continue;
          auto hp_id = static_cast<loom::IdIndex>(*hp);
          if (sp < state.swPortToHwPort.size()) {
            state.swPortToHwPort[sp] = hp_id;
            if (hp_id < state.hwPortToSwPorts.size())
              state.hwPortToSwPorts[hp_id].push_back(sp);
          }
        }
      }
      if (auto *rt = root->getObject("routes")) {
        for (auto &kv : *rt) {
          unsigned se;
          if (llvm::StringRef(kv.first).getAsInteger(10, se))
            continue;
          auto *entry = kv.second.getAsObject();
          if (!entry)
            continue;
          auto *hw_path = entry->getArray("hwPath");
          if (!hw_path)
            continue;
          llvm::SmallVector<loom::IdIndex, 8> path;
          for (auto &hop : *hw_path) {
            auto *h = hop.getAsObject();
            if (!h)
              continue;
            auto s = h->getInteger("src");
            auto d = h->getInteger("dst");
            if (s && d) {
              path.push_back(static_cast<loom::IdIndex>(*s));
              path.push_back(static_cast<loom::IdIndex>(*d));
            }
          }
          if (se < state.swEdgeToHwPaths.size())
            state.swEdgeToHwPaths[se] = path;
        }
      }

      // Generate mapped visualization DOTs.
      std::string overlay_dot =
          loom::viz::exportMappedOverlayDot(dfg, adg, state, viz_opts);
      std::string dfg_dot =
          loom::viz::exportMappedDFGDot(dfg, state, viz_opts);
      std::string adg_dot =
          loom::viz::exportMappedADGDot(dfg, adg, state, viz_opts);

      // Build mapping JSON for the HTML viewer.
      std::ostringstream mj;
      mj << "{\"swToHw\":{";
      bool mj_first = true;
      for (size_t i = 0; i < state.swNodeToHwNode.size(); ++i) {
        if (state.swNodeToHwNode[i] == loom::INVALID_ID)
          continue;
        if (!mj_first) mj << ",";
        mj << "\"sw_" << i << "\":\"hw_"
           << state.swNodeToHwNode[i] << "\"";
        mj_first = false;
      }
      mj << "},\"hwToSw\":{";
      mj_first = true;
      for (size_t i = 0; i < state.hwNodeToSwNodes.size(); ++i) {
        bool has_valid = false;
        for (auto sid : state.hwNodeToSwNodes[i])
          if (sid != loom::INVALID_ID) has_valid = true;
        if (!has_valid) continue;
        if (!mj_first) mj << ",";
        mj << "\"hw_" << i << "\":[";
        bool inner_first = true;
        for (auto sid : state.hwNodeToSwNodes[i]) {
          if (sid == loom::INVALID_ID) continue;
          if (!inner_first) mj << ",";
          mj << "\"sw_" << sid << "\"";
          inner_first = false;
        }
        mj << "]";
        mj_first = false;
      }
      mj << "},\"routes\":{}}";

      std::string html = loom::viz::generateMappedHTML(
          overlay_dot, dfg_dot, adg_dot, mj.str(), "{}", "Mapped View");

      std::error_code ec;
      llvm::raw_fd_ostream out(parsed.output_path, ec, llvm::sys::fs::OF_Text);
      if (ec) {
        llvm::errs() << "error: cannot write: " << parsed.output_path << "\n";
        return 1;
      }
      out << html;
      out.flush();
      return 0;
    }
  }

  // ADG mode: validation-only or mapper invocation.
  if (!parsed.adg_path.empty()) {
    bool has_sources = !parsed.inputs.empty();
    bool has_handshake = !parsed.handshake_input.empty();

    // Mode 1: Validation only (--adg without sources and without --handshake-input).
    if (!has_sources && !has_handshake) {
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

    // Mode 2: Mapper invocation (--adg with sources or --handshake-input).
    if (parsed.output_path.empty()) {
      llvm::errs() << "error: -o is required for mapper mode\n";
      return 1;
    }

    if (!EnsureOutputDirectory(parsed.output_path))
      return 1;

    // Set up shared MLIR context with all required dialects.
    mlir::MLIRContext mapper_context;
    mlir::DialectRegistry mapper_registry;
    mlir::func::registerInlinerExtension(mapper_registry);
    mapper_context.appendDialectRegistry(mapper_registry);
    mapper_context.getDiagEngine().registerHandler(
        [](mlir::Diagnostic &diag) {
          diag.print(llvm::errs());
          llvm::errs() << "\n";
          return mlir::success();
        });
    mapper_context.getOrLoadDialect<mlir::LLVM::LLVMDialect>();
    mapper_context.getOrLoadDialect<mlir::DLTIDialect>();
    mapper_context.getOrLoadDialect<mlir::arith::ArithDialect>();
    mapper_context.getOrLoadDialect<mlir::cf::ControlFlowDialect>();
    mapper_context.getOrLoadDialect<mlir::func::FuncDialect>();
    mapper_context.getOrLoadDialect<mlir::math::MathDialect>();
    mapper_context.getOrLoadDialect<mlir::memref::MemRefDialect>();
    mapper_context.getOrLoadDialect<mlir::scf::SCFDialect>();
    mapper_context.getOrLoadDialect<mlir::ub::UBDialect>();
    mapper_context.getOrLoadDialect<loom::dataflow::DataflowDialect>();
    mapper_context.getOrLoadDialect<loom::fabric::FabricDialect>();
    mapper_context.getOrLoadDialect<circt::handshake::HandshakeDialect>();
    mapper_context.getOrLoadDialect<circt::esi::ESIDialect>();
    mapper_context.getOrLoadDialect<circt::hw::HWDialect>();
    mapper_context.getOrLoadDialect<circt::seq::SeqDialect>();

    // Parse the ADG (fabric MLIR).
    mlir::ParserConfig adg_parser_config(&mapper_context);
    auto adg_module = mlir::parseSourceFile<mlir::ModuleOp>(
        parsed.adg_path, adg_parser_config);
    if (!adg_module) {
      llvm::errs() << "error: failed to parse ADG: " << parsed.adg_path << "\n";
      return 1;
    }
    if (failed(mlir::verify(*adg_module))) {
      llvm::errs() << "error: ADG verification failed: " << parsed.adg_path << "\n";
      return 1;
    }

    // Obtain Handshake MLIR: either compile from sources or load from file.
    mlir::OwningOpRef<mlir::ModuleOp> handshake_module;
    if (has_handshake) {
      // Load pre-compiled Handshake MLIR directly.
      handshake_module = mlir::parseSourceFile<mlir::ModuleOp>(
          parsed.handshake_input, adg_parser_config);
      if (!handshake_module) {
        llvm::errs() << "error: failed to parse handshake input: "
                     << parsed.handshake_input << "\n";
        return 1;
      }
      if (failed(mlir::verify(*handshake_module))) {
        llvm::errs() << "error: handshake input verification failed\n";
        return 1;
      }
    }
    // Source compilation path would go here (compile sources -> handshake MLIR).
    // For now, source compilation reuses the standard pipeline below and then
    // the handshake_module is set from the resulting module. This is handled
    // by falling through to the standard compilation path when has_sources is
    // true, so for the --handshake-input path we proceed directly to mapping.

    if (has_sources) {
      // Compile sources through the standard frontend pipeline to produce
      // Handshake MLIR, then continue with the mapper below.

      // Derive base path for Stage A output files.
      std::string sa_base = parsed.output_path;
      {
        llvm::StringRef bp(sa_base);
        if (bp.ends_with(".config.bin"))
          sa_base = bp.drop_back(sizeof(".config.bin") - 1).str();
        else if (bp.ends_with(".llvm.ll"))
          sa_base = bp.drop_back(sizeof(".llvm.ll") - 1).str();
        else if (bp.ends_with(".ll"))
          sa_base = bp.drop_back(sizeof(".ll") - 1).str();
      }
      std::string sa_ll_path = sa_base + ".llvm.ll";
      mlir::OpPrintingFlags sa_print_flags;
      sa_print_flags.enableDebugInfo(true, false);

      llvm::InitializeNativeTarget();
      llvm::InitializeNativeTargetAsmPrinter();
      llvm::InitializeNativeTargetAsmParser();

      std::string src_exe =
          llvm::sys::fs::getMainExecutable(argv[0],
                                           reinterpret_cast<void *>(&main));
      std::vector<std::string> src_drv_args =
          BuildDriverArgs(parsed.driver_args);
      if (!HasResourceDirArg(src_drv_args)) {
        llvm::SmallString<256> rdir =
            llvm::sys::path::parent_path(src_exe);
        rdir = llvm::sys::path::parent_path(rdir);
        llvm::sys::path::append(rdir, "lib", "clang");
        src_drv_args.push_back("-resource-dir=" + rdir.str().str());
      }

      clang::DiagnosticOptions src_diag_opts;
      auto src_diag_client =
          std::make_unique<clang::TextDiagnosticPrinter>(
              llvm::errs(), src_diag_opts);
      auto src_diags = clang::CompilerInstance::createDiagnostics(
          *llvm::vfs::getRealFileSystem(), src_diag_opts,
          src_diag_client.get(), /*ShouldOwnClient=*/false);

      clang::driver::Driver src_driver(
          src_exe, llvm::sys::getDefaultTargetTriple(), *src_diags);
      src_driver.setTitle("loom");
      src_driver.setCheckInputsExist(true);

      std::vector<const char *> src_cmdline;
      src_cmdline.reserve(1 + src_drv_args.size() + parsed.inputs.size());
      src_cmdline.push_back(src_exe.c_str());
      for (const auto &a : src_drv_args)
        src_cmdline.push_back(a.c_str());
      for (const auto &inp : parsed.inputs)
        src_cmdline.push_back(inp.c_str());

      std::unique_ptr<clang::driver::Compilation> src_comp(
          src_driver.BuildCompilation(src_cmdline));
      if (!src_comp) {
        llvm::errs() << "error: failed to build compilation\n";
        return 1;
      }

      llvm::LLVMContext src_llvm_ctx;
      std::unique_ptr<llvm::Module> src_linked;
      unsigned src_count = 0;

      for (const auto &job : src_comp->getJobs()) {
        if (!IsCC1Command(job))
          continue;
        auto inv = std::make_shared<clang::CompilerInvocation>();
        if (!clang::CompilerInvocation::CreateFromArgs(
                *inv, job.getArguments(), *src_diags, argv[0])) {
          llvm::errs() << "error: failed to build compiler invocation\n";
          return 1;
        }
        if (inv->getFrontendOpts().Inputs.empty()) {
          llvm::errs() << "error: missing input in compiler invocation\n";
          return 1;
        }
        inv->getCodeGenOpts().setDebugInfo(
            llvm::codegenoptions::FullDebugInfo);

        auto src_mod = CompileInvocation(inv, src_llvm_ctx);
        if (!src_mod) {
          llvm::errs() << "error: failed to compile source\n";
          return 1;
        }
        if (!src_linked) {
          src_linked = std::move(src_mod);
          src_count++;
          continue;
        }
        llvm::Linker linker(*src_linked);
        if (linker.linkInModule(std::move(src_mod))) {
          llvm::errs() << "error: failed to link source module\n";
          return 1;
        }
        src_count++;
      }

      if (src_count == 0) {
        llvm::errs() << "error: no compilation jobs generated\n";
        return 1;
      }

      std::string src_vfy_err;
      llvm::raw_string_ostream src_vfy_os(src_vfy_err);
      if (llvm::verifyModule(*src_linked, &src_vfy_os)) {
        llvm::errs() << "error: linked module verification failed\n"
                     << src_vfy_os.str() << "\n";
        return 1;
      }

      StripUnsupportedAttributes(*src_linked);
      AnnotationMap src_annot = CollectGlobalAnnotations(*src_linked);

      // Write Stage A: LLVM IR.
      {
        EnsureOutputDirectory(sa_ll_path);
        std::error_code sa_ec;
        llvm::raw_fd_ostream sa_out(sa_ll_path, sa_ec, llvm::sys::fs::OF_Text);
        if (!sa_ec) {
          src_linked->print(sa_out, nullptr);
          sa_out.flush();
        }
      }

      // Translate LLVM IR to MLIR using the mapper context.
      auto src_mlir = mlir::translateLLVMIRToModule(
          std::move(src_linked), &mapper_context,
          /*emitExpensiveWarnings=*/false,
          /*dropDICompositeTypeElements=*/false,
          /*loadAllDialects=*/false);
      if (!src_mlir) {
        llvm::errs() << "error: LLVM IR to MLIR translation failed\n";
        return 1;
      }

      ApplySymbolAnnotations(*src_mlir, src_annot);
      ApplyIntrinsicAnnotations(*src_mlir);
      ApplyLoopMarkerAnnotations(*src_mlir);

      if (failed(mlir::verify(*src_mlir))) {
        llvm::errs() << "error: MLIR verification failed\n";
        return 1;
      }

      // Write Stage A: MLIR.
      {
        std::string sa_mlir_path = DeriveMlirOutputPath(sa_ll_path);
        EnsureOutputDirectory(sa_mlir_path);
        std::error_code sa_ec;
        llvm::raw_fd_ostream sa_out(sa_mlir_path, sa_ec, llvm::sys::fs::OF_Text);
        if (!sa_ec) {
          src_mlir->print(sa_out, sa_print_flags);
          sa_out.flush();
        }
      }

      // Run LLVMToSCF pipeline.
      mlir::PassManager scf_pm(&mapper_context);
      scf_pm.addPass(loom::createConvertLLVMToSCFPass());
      scf_pm.addPass(mlir::createCanonicalizerPass());
      scf_pm.addPass(mlir::createCSEPass());
      scf_pm.addPass(mlir::createMem2Reg());
      scf_pm.addPass(mlir::createCanonicalizerPass());
      scf_pm.addPass(mlir::createCSEPass());
      scf_pm.addPass(mlir::createLiftControlFlowToSCFPass());
      scf_pm.addPass(mlir::createLoopInvariantCodeMotionPass());
      scf_pm.addPass(mlir::createCanonicalizerPass());
      scf_pm.addPass(mlir::createCSEPass());
      scf_pm.addPass(loom::createUpliftWhileToForPass());
      scf_pm.addPass(mlir::createCanonicalizerPass());
      scf_pm.addPass(mlir::createCSEPass());
      scf_pm.addPass(loom::createAttachLoopAnnotationsPass());
      scf_pm.addPass(loom::createMarkWhileStreamablePass());
      if (failed(scf_pm.run(*src_mlir))) {
        llvm::errs() << "error: LLVMToSCF conversion failed\n";
        return 1;
      }

      // Write Stage A: SCF MLIR.
      {
        std::string sa_scf_path = DeriveScfOutputPath(sa_ll_path);
        EnsureOutputDirectory(sa_scf_path);
        std::error_code sa_ec;
        llvm::raw_fd_ostream sa_out(sa_scf_path, sa_ec, llvm::sys::fs::OF_Text);
        if (!sa_ec) {
          src_mlir->print(sa_out, sa_print_flags);
          sa_out.flush();
        }
      }

      // Run SCFToHandshake pipeline.
      mlir::PassManager hs_pm(&mapper_context);
      hs_pm.addPass(loom::createSCFToHandshakeDataflowPass());
      hs_pm.addPass(mlir::createCanonicalizerPass());
      hs_pm.addPass(mlir::createCSEPass());
      if (failed(hs_pm.run(*src_mlir))) {
        llvm::errs() << "error: SCFToHandshake conversion failed\n";
        return 1;
      }

      if (failed(mlir::verify(*src_mlir))) {
        llvm::errs() << "error: Handshake verification failed\n";
        return 1;
      }

      // Write Stage A: Handshake MLIR.
      {
        std::string sa_hs_path = DeriveHandshakeOutputPath(sa_ll_path);
        EnsureOutputDirectory(sa_hs_path);
        std::error_code sa_ec;
        llvm::raw_fd_ostream sa_out(sa_hs_path, sa_ec, llvm::sys::fs::OF_Text);
        if (!sa_ec) {
          src_mlir->print(sa_out, sa_print_flags);
          sa_out.flush();
        }
      }

      handshake_module = std::move(src_mlir);
    }

    // Find the handshake::FuncOp in the handshake module for DFG extraction.
    circt::handshake::FuncOp handshake_func;
    handshake_module->walk([&](circt::handshake::FuncOp func) {
      llvm::StringRef name = func.getName();
      bool isEsi = name.ends_with("_esi");
      if (!handshake_func ||
          (!isEsi && handshake_func.getName().ends_with("_esi")))
        handshake_func = func;
    });
    if (!handshake_func) {
      llvm::errs() << "error: no handshake.func found in input\n";
      return 1;
    }

    // Find the fabric::ModuleOp in the ADG module.
    loom::fabric::ModuleOp fabric_mod;
    adg_module->walk([&](loom::fabric::ModuleOp mod) {
      if (!fabric_mod)
        fabric_mod = mod;
    });
    if (!fabric_mod) {
      llvm::errs() << "error: no fabric.module found in ADG\n";
      return 1;
    }

    // Extract DFG from Handshake IR.
    loom::DFGBuilder dfg_builder;
    loom::Graph dfg = dfg_builder.build(handshake_func);

    // Flatten ADG from Fabric IR.
    loom::ADGFlattener adg_flattener;
    loom::Graph adg = adg_flattener.flatten(fabric_mod);

    // Run technology mapping.
    loom::TechMapper tech_mapper;
    loom::CandidateSet candidates = tech_mapper.map(dfg, adg);

    // Set up mapper options from CLI flags.
    loom::Mapper::Options mapper_opts;
    mapper_opts.budgetSeconds = parsed.mapper_budget;
    mapper_opts.seed = parsed.mapper_seed;
    mapper_opts.profile = parsed.mapper_profile;

    // Run the PnR mapper.
    loom::Mapper mapper;
    loom::Mapper::Result mapper_result = mapper.run(dfg, adg, mapper_opts);

    if (!mapper_result.success) {
      llvm::errs() << "error: mapping failed\n";
      if (!mapper_result.diagnostics.empty())
        llvm::errs() << mapper_result.diagnostics << "\n";
      return 1;
    }

    // Generate configuration output files.
    loom::ConfigGen config_gen;
    std::string base_path = parsed.output_path;
    // Strip extension for base path derivation.
    llvm::StringRef base_ref(base_path);
    if (base_ref.ends_with(".config.bin"))
      base_path = base_ref.drop_back(sizeof(".config.bin") - 1).str();
    else if (base_ref.ends_with(".llvm.ll"))
      base_path = base_ref.drop_back(sizeof(".llvm.ll") - 1).str();
    else if (base_ref.ends_with(".ll"))
      base_path = base_ref.drop_back(sizeof(".ll") - 1).str();

    if (!config_gen.generate(mapper_result.state, dfg, adg, base_path,
                             parsed.dump_mapping, parsed.mapper_profile,
                             parsed.mapper_seed)) {
      llvm::errs() << "error: configuration generation failed\n";
      return 1;
    }

    llvm::outs() << "mapping succeeded: " << base_path << ".config.bin\n";

    // Generate visualization HTML if --dump-viz is set.
    if (parsed.dump_viz) {
      loom::viz::DOTOptions viz_opts;
      if (parsed.viz_mode == "detailed")
        viz_opts.mode = loom::viz::DOTMode::Detailed;

      std::string overlay_dot =
          loom::viz::exportMappedOverlayDot(dfg, adg, mapper_result.state,
                                            viz_opts);
      std::string dfg_dot =
          loom::viz::exportMappedDFGDot(dfg, mapper_result.state, viz_opts);
      std::string adg_dot =
          loom::viz::exportMappedADGDot(dfg, adg, mapper_result.state,
                                        viz_opts);

      // Build mapping JSON for the HTML viewer.
      std::ostringstream mj;
      mj << "{ \"swToHw\": {";
      bool mj_first = true;
      for (size_t i = 0; i < mapper_result.state.swNodeToHwNode.size(); ++i) {
        if (mapper_result.state.swNodeToHwNode[i] == loom::INVALID_ID)
          continue;
        if (!mj_first) mj << ",";
        mj << "\"sw_" << i << "\":\"hw_"
           << mapper_result.state.swNodeToHwNode[i] << "\"";
        mj_first = false;
      }
      mj << "}, \"hwToSw\": {";
      mj_first = true;
      for (size_t i = 0; i < mapper_result.state.hwNodeToSwNodes.size(); ++i) {
        bool has_valid = false;
        for (auto sid : mapper_result.state.hwNodeToSwNodes[i])
          if (sid != loom::INVALID_ID) has_valid = true;
        if (!has_valid) continue;
        if (!mj_first) mj << ",";
        mj << "\"hw_" << i << "\":[";
        bool inner_first = true;
        for (auto sid : mapper_result.state.hwNodeToSwNodes[i]) {
          if (sid == loom::INVALID_ID) continue;
          if (!inner_first) mj << ",";
          mj << "\"sw_" << sid << "\"";
          inner_first = false;
        }
        mj << "]";
        mj_first = false;
      }
      mj << "}, \"routes\": {} }";

      std::string mapped_html = loom::viz::generateMappedHTML(
          overlay_dot, dfg_dot, adg_dot, mj.str(), "{}", "Mapped View");

      std::string viz_path = base_path + ".mapped.html";
      std::error_code viz_ec;
      llvm::raw_fd_ostream viz_out(viz_path, viz_ec, llvm::sys::fs::OF_Text);
      if (viz_ec) {
        llvm::errs() << "warning: cannot write viz: " << viz_path << "\n";
      } else {
        viz_out << mapped_html;
        viz_out.flush();
        llvm::outs() << "visualization: " << viz_path << "\n";
      }

      // DFG visualization.
      {
        std::string dfg_only_dot = loom::viz::exportDFGDot(dfg, viz_opts);
        std::string dfg_html =
            loom::viz::generateHTML(dfg_only_dot, "DFG Visualization");
        std::string dfg_path = base_path + ".dfg.html";
        std::error_code dfg_ec;
        llvm::raw_fd_ostream dfg_out(dfg_path, dfg_ec, llvm::sys::fs::OF_Text);
        if (dfg_ec) {
          llvm::errs() << "warning: cannot write viz: " << dfg_path << "\n";
        } else {
          dfg_out << dfg_html;
          dfg_out.flush();
          llvm::outs() << "visualization: " << dfg_path << "\n";
        }
      }

      // ADG visualization.
      {
        std::string adg_only_dot = loom::viz::exportADGDot(adg, viz_opts);
        std::string adg_html =
            loom::viz::generateHTML(adg_only_dot, "ADG Visualization");
        std::string adg_path = base_path + ".adg.html";
        std::error_code adg_ec;
        llvm::raw_fd_ostream adg_out(adg_path, adg_ec, llvm::sys::fs::OF_Text);
        if (adg_ec) {
          llvm::errs() << "warning: cannot write viz: " << adg_path << "\n";
        } else {
          adg_out << adg_html;
          adg_out.flush();
          llvm::outs() << "visualization: " << adg_path << "\n";
        }
      }
    }

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
