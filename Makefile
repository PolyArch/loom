# Wrapper Makefile for loom.
#
# Targets:
#   make llvm   - configure and build externals/llvm (LLVM + MLIR, ccache on)
#   make loom   - configure and build the loom project against the above
#   make test   - run lit FileCheck tests (target: check-fabric)
#   make clean  - remove loom build tree only
#   make distclean - remove both loom and llvm build trees

ROOT        := $(abspath $(CURDIR))
LLVM_SRC    := $(ROOT)/externals/llvm/llvm
LLVM_BUILD  := $(ROOT)/externals/llvm/build
LOOM_BUILD  := $(ROOT)/build

MLIR_DIR    := $(LLVM_BUILD)/lib/cmake/mlir
LLVM_DIR    := $(LLVM_BUILD)/lib/cmake/llvm
LLVM_LIT    := $(LLVM_BUILD)/bin/llvm-lit

JOBS        ?= $(shell nproc)

.PHONY: all llvm loom test clean distclean

all: loom

llvm: $(LLVM_BUILD)/build.ninja
	cmake --build $(LLVM_BUILD) -j$(JOBS)

$(LLVM_BUILD)/build.ninja:
	cmake -G Ninja -S $(LLVM_SRC) -B $(LLVM_BUILD) \
	  -DCMAKE_BUILD_TYPE=Release \
	  -DLLVM_ENABLE_PROJECTS="mlir" \
	  -DLLVM_TARGETS_TO_BUILD="host" \
	  -DLLVM_ENABLE_ASSERTIONS=ON \
	  -DLLVM_ENABLE_RTTI=ON \
	  -DLLVM_INSTALL_UTILS=ON \
	  -DLLVM_CCACHE_BUILD=ON \
	  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
	  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
	  -DBUILD_SHARED_LIBS=OFF \
	  -DLLVM_BUILD_LLVM_DYLIB=ON \
	  -DLLVM_LINK_LLVM_DYLIB=ON

loom: $(LOOM_BUILD)/build.ninja
	cmake --build $(LOOM_BUILD) -j$(JOBS)

$(LOOM_BUILD)/build.ninja:
	@if [ ! -f "$(MLIR_DIR)/MLIRConfig.cmake" ]; then \
	  echo "error: MLIR not built. Run 'make llvm' first." >&2; exit 1; \
	fi
	cmake -G Ninja -S $(ROOT) -B $(LOOM_BUILD) \
	  -DCMAKE_BUILD_TYPE=Release \
	  -DMLIR_DIR=$(MLIR_DIR) \
	  -DLLVM_DIR=$(LLVM_DIR) \
	  -DLLVM_EXTERNAL_LIT=$(LLVM_LIT) \
	  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
	  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache

test: loom
	cmake --build $(LOOM_BUILD) -j$(JOBS) --target check-fabric

clean:
	rm -rf $(LOOM_BUILD)

distclean:
	rm -rf $(LOOM_BUILD) $(LLVM_BUILD)
