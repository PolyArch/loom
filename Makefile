# Wrapper Makefile for loom.
#
# All worktrees (the main one plus any `git worktree add` linked
# checkouts) share a single LLVM source tree and build directory living
# under the main worktree (the one that owns `.git/`). Linked
# worktrees keep their own loom build directory, but reference the
# shared LLVM artifacts through cmake (-DMLIR_DIR / -DLLVM_DIR /
# -DLLVM_EXTERNAL_LIT).
#
# Targets:
#   make llvm      - build externals/llvm in the MAIN worktree under flock
#   make loom      - build this worktree's loom build (auto-builds llvm
#                    if the shared MLIRConfig.cmake is missing)
#   make test      - run lit FileCheck tests (target: check-fabric)
#   make clean     - remove this worktree's loom build only
#   make distclean - main worktree: remove both loom and shared llvm
#                    builds. Linked worktree: remove only this loom
#                    build (the shared llvm is left alone).

ROOT          := $(abspath $(CURDIR))

# Resolve the main worktree path. `git rev-parse --git-common-dir`
# returns <main>/.git regardless of which worktree we're in; the
# parent of that is the main worktree. Falls back to $(ROOT) when
# we are not inside a git checkout.
MAIN_WORKTREE := $(shell git -C $(ROOT) rev-parse --path-format=absolute --git-common-dir 2>/dev/null | xargs -r dirname)
ifeq ($(strip $(MAIN_WORKTREE)),)
  MAIN_WORKTREE := $(ROOT)
endif

# Shared LLVM artifacts (anchored on the main worktree).
LLVM_SRC      := $(MAIN_WORKTREE)/externals/llvm/llvm
LLVM_BUILD    := $(MAIN_WORKTREE)/externals/llvm/build
LLVM_LOCK     := $(MAIN_WORKTREE)/externals/llvm/.build.lock

# Per-worktree loom build.
LOOM_BUILD    := $(ROOT)/build

MLIR_DIR      := $(LLVM_BUILD)/lib/cmake/mlir
LLVM_DIR      := $(LLVM_BUILD)/lib/cmake/llvm
LLVM_LIT      := $(LLVM_BUILD)/bin/llvm-lit

JOBS          ?= $(shell nproc)
FLOCK         ?= flock
PYTHON        ?= python3

# Whether the current invocation is running in the main worktree.
IS_MAIN       := $(if $(filter $(MAIN_WORKTREE),$(ROOT)),1,)

.PHONY: all llvm _llvm_locked loom test clean distclean

all: loom

# Public llvm target: serialised across all worktrees via flock on the
# shared $(LLVM_LOCK) file. The lock file descriptor is inherited by
# the recursive make invocation, so the lock is held for the full
# configure + build window.
llvm:
	@command -v $(FLOCK) >/dev/null 2>&1 || { \
	  echo "error: $(FLOCK) not found; install util-linux for cross-worktree LLVM serialisation" >&2; \
	  exit 1; \
	}
	@mkdir -p $(dir $(LLVM_LOCK))
	@if [ "$(MAIN_WORKTREE)" != "$(ROOT)" ]; then \
	  echo "info: building shared LLVM under main worktree $(MAIN_WORKTREE)" >&2; \
	fi
	@$(FLOCK) "$(LLVM_LOCK)" $(MAKE) -f $(firstword $(MAKEFILE_LIST)) _llvm_locked

# Internal target: actually configure / build LLVM. Always invoked via
# `flock $(LLVM_LOCK)` from the public `llvm` target so concurrent
# worktrees serialise on the shared LLVM build.
_llvm_locked: $(LLVM_BUILD)/build.ninja
	cmake --build $(LLVM_BUILD) -j$(JOBS)

$(LLVM_BUILD)/build.ninja:
	cmake -G Ninja -S $(LLVM_SRC) -B $(LLVM_BUILD) \
	  -DCMAKE_BUILD_TYPE=Release \
	  -DCMAKE_C_COMPILER=clang \
	  -DCMAKE_CXX_COMPILER=clang++ \
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
	  echo "info: shared MLIR not found at $(LLVM_BUILD); building it now" >&2; \
	  $(MAKE) -f $(firstword $(MAKEFILE_LIST)) llvm; \
	fi
	cmake -G Ninja -S $(ROOT) -B $(LOOM_BUILD) \
	  -DCMAKE_BUILD_TYPE=Release \
	  -DCMAKE_C_COMPILER=clang \
	  -DCMAKE_CXX_COMPILER=clang++ \
	  -DMLIR_DIR=$(MLIR_DIR) \
	  -DLLVM_DIR=$(LLVM_DIR) \
	  -DLLVM_EXTERNAL_LIT=$(LLVM_LIT) \
	  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
	  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache

test: loom
	@bash -o pipefail -c 'LIT_OPTS="-sv --time-tests $(LIT_OPTS)" \
	  cmake --build "$(LOOM_BUILD)" -j"$(JOBS)" --target check-fabric 2>&1 \
	  | "$(PYTHON)" "$(ROOT)/test/lit_top_slowest.py"'

clean:
	rm -rf $(LOOM_BUILD)

distclean:
	rm -rf $(LOOM_BUILD)
ifneq ($(IS_MAIN),)
	rm -rf $(LLVM_BUILD)
else
	@echo "info: distclean from linked worktree only removes $(LOOM_BUILD); shared LLVM at $(LLVM_BUILD) preserved" >&2
endif
