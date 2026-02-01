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
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/VirtualFileSystem.h"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace {

struct ParsedArgs {
  std::vector<std::string> inputs;
  std::vector<std::string> driver_args;
  std::string output_path;
  bool show_help = false;
  bool show_version = false;
  bool had_error = false;
};

void PrintUsage(llvm::StringRef prog) {
  llvm::outs() << "Usage: " << prog
               << " [options] <sources...> -o <output.ll>\n";
  llvm::outs() << "\n";
  llvm::outs() << "Compile and link C++ sources into a single LLVM IR file.\n";
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

  bool has_dash_x = false;
  bool has_compile_only = false;
  for (const auto &arg : args) {
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

  return 0;
}
