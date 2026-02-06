import lit.formats
import os
import pathlib

config.name = "Loom Fabric TDD"
config.test_format = lit.formats.ShTest(execute_external=True)
config.suffixes = ['.mlir']

root = pathlib.Path(__file__).resolve().parents[3]
build_bin = str(root / "build" / "bin")
llvm_bin = str(root / "build" / "externals" / "llvm-project" / "llvm" / "bin")

config.test_source_root = str(pathlib.Path(__file__).resolve().parent)

config.environment = dict(os.environ)
config.environment['PATH'] = f"{build_bin}:{llvm_bin}:" + os.environ.get('PATH', '')
