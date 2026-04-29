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

tool_dirs = [
    os.path.join(config.loom_obj_root, "tools", "loom"),
    os.path.join(config.loom_obj_root, "tools", "loom-candidate-dump"),
    os.path.join(config.loom_obj_root, "tools", "loom-config-test"),
    os.path.join(config.loom_obj_root, "tools", "loom-hwsg-test"),
    os.path.join(config.loom_obj_root, "tools", "loom-parallel-test"),
    os.path.join(config.loom_obj_root, "tools", "loom-synth-config-test"),
    os.path.join(config.loom_obj_root, "tools", "loom-template-dump"),
    os.path.join(config.loom_obj_root, "bin"),
    config.llvm_tools_dir,
]
tools = [
    "loom",
    "loom-candidate-dump",
    "loom-config-test",
    "loom-hwsg-test",
    "loom-parallel-test",
    "loom-synth-config-test",
    "loom-template-dump",
    "mlir-opt",
]
llvm_config.add_tool_substitutions(tools, tool_dirs)
