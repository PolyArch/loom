//===-- loom_args.cpp - Argument parsing for Loom driver --------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include "loom_args.h"

#include "clang/Basic/Version.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

namespace loom {
namespace tool {

std::string DeriveConfigBinPath(llvm::StringRef output_path) {
  llvm::SmallString<256> path(output_path);
  llvm::sys::path::replace_extension(path, ".config.bin");
  return std::string(path);
}

std::string DeriveAddrHeaderPath(llvm::StringRef output_path) {
  llvm::SmallString<256> base(output_path);
  llvm::sys::path::replace_extension(base, "");
  return (base + "_addr.h").str();
}

std::string DeriveMappingJsonPath(llvm::StringRef output_path) {
  llvm::SmallString<256> path(output_path);
  llvm::sys::path::replace_extension(path, ".mapping.json");
  return std::string(path);
}

void PrintUsage(llvm::StringRef prog) {
  llvm::outs() << "Usage: " << prog
               << " [options] <sources...> -o <output.llvm.ll>\n";
  llvm::outs() << "       " << prog
               << " --adg <file.fabric.mlir>\n";
  llvm::outs() << "       " << prog
               << " --adg <fabric.mlir> <sources...> -o <output>\n";
  llvm::outs() << "       " << prog
               << " --adg <fabric.mlir> --handshake-input <file> -o <output>\n";
  llvm::outs() << "       " << prog
               << " --as-clang [clang-options...]\n";
  llvm::outs() << "\n";
  llvm::outs() << "Compile and link C++ sources into a single LLVM IR file "
               << "and emit LLVM dialect MLIR.\n";
  llvm::outs() << "The MLIR output path is derived from -o by replacing "
               << ".llvm.ll or .ll with .mlir.\n";
  llvm::outs() << "\n";
  llvm::outs() << "ADG validation mode (--adg without sources):\n";
  llvm::outs() << "  Parse a fabric MLIR file, run semantic verification, "
               << "and exit 0 (valid) or 1 (errors).\n";
  llvm::outs() << "\n";
  llvm::outs() << "Mapper mode (--adg with sources or --handshake-input):\n";
  llvm::outs() << "  Compile sources, extract DFG, run place-and-route, "
               << "and emit configuration.\n";
  llvm::outs() << "\n";
  llvm::outs() << "Mapper options:\n";
  llvm::outs() << "  --mapper-budget <seconds>  Search time limit (default: 60)\n";
  llvm::outs() << "  --mapper-seed <int>        Deterministic seed (default: 0)\n";
  llvm::outs() << "  --mapper-profile <name>    Weight profile (default: balanced)\n";
  llvm::outs() << "  --dump-mapping             Emit .mapping.json report\n";
  llvm::outs() << "  --handshake-input <file>   Use pre-compiled Handshake MLIR\n";
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
      if (arg == "--handshake-input") {
        if (i + 1 >= argc) {
          llvm::errs() << "error: --handshake-input requires a path\n";
          parsed.had_error = true;
          break;
        }
        parsed.handshake_input = argv[++i];
        continue;
      }
      if (arg == "--mapper-budget") {
        if (i + 1 >= argc) {
          llvm::errs() << "error: --mapper-budget requires a value\n";
          parsed.had_error = true;
          break;
        }
        parsed.mapper_budget = std::stod(argv[++i]);
        continue;
      }
      if (arg == "--mapper-seed") {
        if (i + 1 >= argc) {
          llvm::errs() << "error: --mapper-seed requires a value\n";
          parsed.had_error = true;
          break;
        }
        parsed.mapper_seed = std::stoi(argv[++i]);
        continue;
      }
      if (arg == "--mapper-profile") {
        if (i + 1 >= argc) {
          llvm::errs() << "error: --mapper-profile requires a value\n";
          parsed.had_error = true;
          break;
        }
        parsed.mapper_profile = argv[++i];
        continue;
      }
      if (arg == "--dump-mapping") {
        parsed.dump_mapping = true;
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

} // namespace tool
} // namespace loom
