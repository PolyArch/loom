SHELL := /bin/sh

.PHONY: all init build rebuild test clean purge

all: rebuild test

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
	cmake -S . -B build -G Ninja -DLLVM_ENABLE_PROJECTS="clang;mlir" -DLLVM_TARGETS_TO_BUILD=host; \
	ninja -C build loom

rebuild: init
	@set -e; \
	cmake -S . -B build -G Ninja -DLLVM_ENABLE_PROJECTS="clang;mlir" -DLLVM_TARGETS_TO_BUILD=host; \
	ninja -C build clang mlir-opt mlir-translate loom FileCheck not

test:
	@set -e; \
	ninja -C build check-loom

clean:
	@set -e; \
	ninja -C build clean-loom

purge:
	@set -e; \
	rm -rf build
