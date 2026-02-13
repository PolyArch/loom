SHELL := /bin/sh

.PHONY: all init build rebuild check clean purge

all: rebuild check

init:
	@set -e; \
	if [ -e externals/llvm-project/.git ]; then \
	  echo "externals/llvm-project already initialized"; \
	else \
	  git -C . submodule update --init --depth 1 --filter=blob:none externals/llvm-project; \
	fi; \
	if [ -e externals/circt/.git ]; then \
	  echo "externals/circt already initialized"; \
	else \
	  git -C . submodule update --init --depth 1 --filter=blob:none externals/circt; \
	fi

build: init
	@set -e; \
	cmake -S . -B build -G Ninja -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_PROJECTS="clang;mlir" -DLLVM_TARGETS_TO_BUILD=host; \
	ninja -C build loom

rebuild: init
	@set -e; \
	cmake -S . -B build -G Ninja -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_PROJECTS="clang;mlir" -DLLVM_TARGETS_TO_BUILD=host; \
	ninja -C build clang mlir-opt mlir-translate loom FileCheck not

check:
	@set -e; \
	ninja -C build check-loom

clean:
	@set -e; \
	if [ -d build ]; then ninja -C build clean-loom; fi
	rm -rf ucli.key *.fsdb novas* verdi* sysProgressP*

purge: clean
	@set -e; \
	rm -rf build .cache
