# Wrapper Makefile for loom.
#
# Path resolution and the worktree edge-case handling (main-worktree
# detection, shared-LLVM serialisation, build-identity stamp tracking,
# stale-loom-build pruning, lock timeouts, NFS warnings) all live in
# scripts/make-worktree.py. This Makefile is a thin dispatcher.
#
# Targets:
#   make doctor    - print resolved paths and run pre-flight checks
#   make llvm      - build externals/llvm under the shared lock
#   make loom      - build this worktree's loom build (auto-builds LLVM
#                    when missing or when its build identity drifted)
#   make test      - run lit FileCheck tests (target: check-fabric)
#   make clean     - remove this worktree's loom build only
#   make distclean - main worktree: remove both loom and shared LLVM
#                    builds. Linked worktree: remove only this loom
#                    build (the shared LLVM is left alone).

ROOT          := $(abspath $(CURDIR))
PYTHON        ?= python3
JOBS          ?= $(shell nproc)
LOCK_TIMEOUT  ?= 1800

WT_SCRIPT     := $(ROOT)/scripts/make-worktree.py
WT            := $(PYTHON) $(WT_SCRIPT) \
                   --root $(ROOT) \
                   --jobs $(JOBS) \
                   --lock-timeout $(LOCK_TIMEOUT)

# LIT_OPTS is consulted by `make test`; export it so the dispatcher's
# environment matches an interactive shell invocation.
export LIT_OPTS
export JOBS

.PHONY: all doctor llvm loom test clean distclean

all: loom

doctor:
	@$(WT) doctor

llvm:
	@$(WT) build-llvm

loom:
	@$(WT) build-loom

test:
	@$(WT) test

clean:
	@$(WT) clean

distclean:
	@$(WT) distclean
