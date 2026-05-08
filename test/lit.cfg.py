import os
import sys

import lit.formats
from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst

config.name = "LOOM"
config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)
config.suffixes = [".mlir"]

config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.loom_obj_root, "test")

config.substitutions.append(("%PATH%", config.environment["PATH"]))
config.substitutions.append(("%shlibext", config.llvm_shlib_ext))
# %python expands to the interpreter running lit, so perf scripts use the
# same Python that drives the test harness.
config.substitutions.append(("%python", sys.executable))

llvm_config.with_system_environment(
    ["HOME", "INCLUDE", "LIB", "TMP", "TEMP",
     "LOOM_BIN", "LOOM_PERF", "LOOM_PERF_CACHE", "LOOM_PERF_TIMEOUT_S"])
llvm_config.use_default_substitutions()

config.excludes = ["lit.cfg.py", "lit.site.cfg.py", "CMakeLists.txt"]

# Perf tests are wall-clock-sensitive and each one claims an exclusive
# core via flock + taskset (see perf_runner.py:claim_exclusive_core).
# Cap the concurrent perf-test count at the number of claimable cores
# so each run lands on a distinct, contention-free core. Honors
# $LOOM_PERF_CORES (same env var perf_runner.py reads). The "perf"
# group is attached per-directory via lit.local.cfg.py under
# test/techmap/perf.
def _perf_parallelism_limit():
    env = os.environ.get("LOOM_PERF_CORES")
    if env:
        cores = [tok.strip() for tok in env.split(",") if tok.strip()]
        if cores:
            return max(1, len(cores))
    try:
        affinity = os.sched_getaffinity(0)
    except AttributeError:
        affinity = set()
    if not affinity:
        return 1
    # Reserve core 0 from the perf pool when there are spare cores
    # (interrupts often pin to it on Linux). Mirrors candidate_perf_cores.
    if len(affinity) > 1 and 0 in affinity:
        affinity = affinity - {0}
    return max(1, len(affinity))


lit_config.parallelism_groups["perf"] = _perf_parallelism_limit()

tool_dirs = [
    os.path.join(config.loom_obj_root, "tools", "loom"),
    os.path.join(config.loom_obj_root, "tools", "loom-cc"),
    os.path.join(config.loom_obj_root, "tools", "loom-alignment-test"),
    os.path.join(config.loom_obj_root, "tools", "loom-candidate-dump"),
    os.path.join(config.loom_obj_root, "tools", "loom-config-test"),
    os.path.join(config.loom_obj_root, "tools", "loom-cost-test"),
    os.path.join(config.loom_obj_root, "tools", "loom-coverage-test"),
    os.path.join(config.loom_obj_root, "tools", "loom-hwsg-test"),
    os.path.join(config.loom_obj_root, "tools", "loom-parallel-test"),
    os.path.join(config.loom_obj_root, "tools", "loom-synth-base-test"),
    os.path.join(config.loom_obj_root, "tools", "loom-synth-config-test"),
    os.path.join(config.loom_obj_root, "tools", "loom-synth-fu-dump"),
    os.path.join(config.loom_obj_root, "tools", "loom-synth-verifier-test"),
    os.path.join(config.loom_obj_root, "tools", "loom-template-dump"),
    os.path.join(config.loom_obj_root, "bin"),
    config.llvm_tools_dir,
]
tools = [
    "loom",
    "loom-alignment-test",
    "loom-candidate-dump",
    "loom-config-test",
    "loom-cost-test",
    "loom-coverage-test",
    "loom-hwsg-test",
    "loom-parallel-test",
    "loom-synth-base-test",
    "loom-synth-config-test",
    "loom-synth-fu-dump",
    "loom-synth-verifier-test",
    "loom-template-dump",
    "mlir-opt",
]
llvm_config.add_tool_substitutions(tools, tool_dirs)

# %loom-c++ and %loom-cc share a "%loom-c" prefix; lit's substitution is
# substring-based, so list the longer pattern first and pin both to the
# built binary path explicitly.
_loom_cc_dir = os.path.join(config.loom_obj_root, "bin")
config.substitutions.insert(
    0, ("%loom-c\\+\\+", os.path.join(_loom_cc_dir, "loom-c++")))
config.substitutions.insert(
    1, ("%loom-cc\\b", os.path.join(_loom_cc_dir, "loom-cc")))
config.substitutions.insert(
    2, ("%objdump-h",
        os.path.join(config.llvm_tools_dir, "llvm-objdump") + " -h"))
