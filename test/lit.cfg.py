import os

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

llvm_config.with_system_environment(["HOME", "INCLUDE", "LIB", "TMP", "TEMP"])
llvm_config.use_default_substitutions()

config.excludes = ["lit.cfg.py", "lit.site.cfg.py", "CMakeLists.txt"]

tool_dirs = [
    os.path.join(config.loom_obj_root, "tools", "loom"),
    os.path.join(config.loom_obj_root, "bin"),
    config.llvm_tools_dir,
]
tools = [
    "loom",
    "mlir-opt",
]
llvm_config.add_tool_substitutions(tools, tool_dirs)
