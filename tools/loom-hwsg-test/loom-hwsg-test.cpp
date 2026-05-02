// Tiny CLI used by lit tests to exercise the loom::common HwShareGroup API.
//
// Subcommands:
//   loom-hwsg-test size
//     Prints "size=<N>" where N is hwShareGroups().size().
//
//   loom-hwsg-test find <op-name>
//     Prints "find <op-name>=<index>" for a multi-member group, or
//     "find <op-name>=none" for a singleton.
//
//   loom-hwsg-test same <op-a> <op-b>
//     Prints "same <op-a> <op-b>=true" or "...=false".
//
// Multiple subcommands may be chained on a single invocation, separated by
// `--`. Each subcommand prints exactly one line. This keeps lit FileCheck
// matching simple and order-stable.

#include "Common/HwShareGroup.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

int usage() {
  ::llvm::errs() << "usage: loom-hwsg-test <subcommand> [args...] "
                    "[-- <subcommand> ...]\n"
                    "  size\n"
                    "  find <op-name>\n"
                    "  same <op-a> <op-b>\n";
  return 2;
}

int runOne(const std::vector<std::string> &args) {
  if (args.empty())
    return usage();
  ::llvm::StringRef cmd = args[0];
  if (cmd == "size") {
    if (args.size() != 1)
      return usage();
    ::llvm::outs() << "size=" << ::loom::common::hwShareGroups().size() << "\n";
    return 0;
  }
  if (cmd == "find") {
    if (args.size() != 2)
      return usage();
    auto idx = ::loom::common::findShareGroup(args[1]);
    ::llvm::outs() << "find " << args[1] << "=";
    if (idx)
      ::llvm::outs() << *idx;
    else
      ::llvm::outs() << "none";
    ::llvm::outs() << "\n";
    return 0;
  }
  if (cmd == "same") {
    if (args.size() != 3)
      return usage();
    bool eq = ::loom::common::sameShareGroup(args[1], args[2]);
    ::llvm::outs() << "same " << args[1] << " " << args[2] << "="
                   << (eq ? "true" : "false") << "\n";
    return 0;
  }
  return usage();
}

} // namespace

int main(int argc, char **argv) {
  std::vector<std::string> bucket;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--") == 0) {
      if (int rc = runOne(bucket); rc != 0)
        return rc;
      bucket.clear();
      continue;
    }
    bucket.emplace_back(argv[i]);
  }
  if (bucket.empty() && argc > 1)
    return 0;
  return runOne(bucket);
}
