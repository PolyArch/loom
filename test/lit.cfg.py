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

# Perf tests are wall-clock-sensitive and pin a single core via taskset.
# Run them one at a time across the suite so concurrent lit workers
# don't load core 0 during the timed window. The "perf" group is
# attached per-directory via lit.local.cfg.py under test/techmap/perf.
lit_config.parallelism_groups["perf"] = 1

tool_dirs = [
    os.path.join(config.loom_obj_root, "tools", "loom"),
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
