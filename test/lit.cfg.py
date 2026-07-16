import os
import sys

import lit.formats
from lit.llvm import llvm_config

config.name = "LOOM"
config.test_format = lit.formats.ShTest()
config.suffixes = [".mlir", ".test"]

config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.loom_obj_root, "test")

config.substitutions.append(("%PATH%", config.environment["PATH"]))
config.substitutions.append(("%shlibext", config.llvm_shlib_ext))
# %python expands to the interpreter running lit.
config.substitutions.append(("%python", sys.executable))

llvm_config.with_system_environment(
    ["HOME", "INCLUDE", "LIB", "TMP", "TEMP",
     "JOBS", "LOOM_TEST_JOBS",
     "LOOM_NATIVE_RUNNER_JOBS"])
llvm_config.use_default_substitutions()

config.excludes = ["lit.cfg.py", "lit.site.cfg.py", "CMakeLists.txt"]

tool_dirs = [
    os.path.join(config.loom_obj_root, "tools", "loom"),
    os.path.join(config.loom_obj_root, "tools", "loom-cc"),
    os.path.join(config.loom_obj_root, "tools", "loom-raise-opt"),
    os.path.join(config.loom_obj_root, "tools", "loom-adg-builder-test"),
    os.path.join(config.loom_obj_root, "tools", "loom-config-test"),
    os.path.join(config.loom_obj_root, "tools", "loom-cost-test"),
    os.path.join(config.loom_obj_root, "tools", "loom-mapping-estimate"),
    os.path.join(config.loom_obj_root, "tools", "loom-dfg-sim"),
    os.path.join(config.loom_obj_root, "tools", "loom-coverage-test"),
    os.path.join(config.loom_obj_root, "tools", "loom-hwsg-test"),
    os.path.join(config.loom_obj_root, "tools", "loom-parallel-test"),
    os.path.join(config.loom_obj_root, "tools", "loom-pnr-map"),
    os.path.join(config.loom_obj_root, "tools", "loom-synth-base-test"),
    os.path.join(config.loom_obj_root, "tools", "loom-synth-config-test"),
    os.path.join(config.loom_obj_root, "tools", "loom-synth-fu-dump"),
    os.path.join(config.loom_obj_root, "bin"),
    config.llvm_tools_dir,
]
tools = [
    "loom",
    "loom-adg-builder-test",
    "loom-config-test",
    "loom-cost-test",
    "loom-mapping-estimate",
    "loom-dfg-sim",
    "loom-coverage-test",
    "loom-hwsg-test",
    "loom-lower",
    "loom-parallel-test",
    "loom-pnr-map",
    "loom-raise",
    "loom-raise-opt",
    "loom-synth-base-test",
    "loom-synth-config-test",
    "loom-synth-fu-dump",
    "mlir-opt",
]
llvm_config.add_tool_substitutions(tools, tool_dirs)

# %loom-c++ / %loom-cc / %loom-raise all share a "%loom-c" or "%loom-r"
# prefix; lit's substitution is substring-based, so list the longest
# patterns first and pin all to the built binary paths explicitly.
_loom_cc_dir = os.path.join(config.loom_obj_root, "bin")
config.substitutions.insert(
    0, ("%loom-c\\+\\+", os.path.join(_loom_cc_dir, "loom-c++")))
config.substitutions.insert(
    1, ("%loom-cc\\b", os.path.join(_loom_cc_dir, "loom-cc")))
config.substitutions.insert(
    2, ("%loom-raise\\b", os.path.join(_loom_cc_dir, "loom-raise")))
config.substitutions.insert(
    3, ("%loom-lower\\b", os.path.join(_loom_cc_dir, "loom-lower")))
config.substitutions.insert(
    4, ("%objdump-h",
        os.path.join(config.llvm_tools_dir, "llvm-objdump") + " -h"))
