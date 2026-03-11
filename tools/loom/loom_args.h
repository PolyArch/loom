//===-- loom_args.h - Argument parsing for Loom driver ----------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#ifndef LOOM_TOOLS_LOOM_ARGS_H
#define LOOM_TOOLS_LOOM_ARGS_H

#include "llvm/ADT/StringRef.h"

#include <string>
#include <vector>

namespace loom {
namespace tool {

struct ParsedArgs {
  std::vector<std::string> inputs;
  std::vector<std::string> driver_args;
  std::string output_path;
  std::string adg_path;
  std::vector<std::string> dfg_paths;
  double mapper_budget = 60.0;
  int mapper_seed = 0;
  std::string mapper_profile = "balanced";
  bool mapper_verbose = false;
  bool mapper_mask_domain = false;
  bool gen_adg = false;
  std::string gen_topology = "mesh"; // "mesh" or "cube"
  unsigned gen_track = 2;
  std::string gen_fifo_mode = "none"; // "none", "single", or "dual"
  unsigned gen_fifo_depth = 2;
  bool gen_fifo_bypassable = false;
  bool gen_temporal = false;
  bool dfg_analyze = false;
  double temporal_threshold = 0.5;
  bool dump_analysis = false;
  bool as_clang = false;
  bool show_help = false;
  bool show_version = false;
  bool had_error = false;
};

std::string DeriveConfigBinPath(llvm::StringRef output_path);
std::string DeriveAddrHeaderPath(llvm::StringRef output_path);
std::string DeriveMapJsonPath(llvm::StringRef output_path);
void PrintUsage(llvm::StringRef prog);
void PrintVersion();
bool IsLinkerFlag(llvm::StringRef arg);
bool LinkerFlagConsumesValue(llvm::StringRef arg);
bool ClangFlagConsumesValue(llvm::StringRef arg);
bool IsDashX(llvm::StringRef arg);
bool HasResourceDirArg(const std::vector<std::string> &args);
ParsedArgs ParseArgs(int argc, char **argv);
std::string DefaultOutputPath(const std::vector<std::string> &inputs);
std::vector<std::string>
BuildDriverArgs(const std::vector<std::string> &user_args);

} // namespace tool
} // namespace loom

#endif // LOOM_TOOLS_LOOM_ARGS_H
