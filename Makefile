# Wrapper Makefile for loom.
#
# Path resolution and the worktree edge-case handling (main-worktree
# detection, primary-owned submodule sources, shared-LLVM serialisation,
# build-identity stamp tracking, stale-loom-build pruning, lock timeouts,
# NFS warnings) all live in scripts/make-worktree.py. This Makefile is a
# thin dispatcher.
#
# Targets:
#   make doctor    - print resolved paths and run pre-flight checks
#   make llvm      - build externals/llvm under the shared lock
#   make circt     - build the shared CIRCT against the shared LLVM under
#                    the same lock (only the libraries the loom build
#                    links against). The shared CIRCT build is owned by
#                    the main worktree's externals; main and linked
#                    invocations alike route to those shared outputs.
#   make loom      - build this worktree's loom build (auto-builds LLVM
#                    when missing or when its build identity drifted;
#                    never builds CIRCT, but offers an already-built
#                    shared CIRCT via -DCIRCT_DIR when one matches)
#   make test      - run lit FileCheck tests (target: check-fabric)
#   make clean     - remove this worktree's loom build only
#   make distclean - main worktree: remove the loom build and both shared
#                    LLVM and CIRCT builds. Linked worktree: remove only
#                    this loom build (shared builds are left alone).

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

.PHONY: all doctor llvm circt loom test clean distclean

all: loom

doctor:
	@$(WT) doctor

llvm:
	@$(WT) build-llvm

circt:
	@$(WT) build-circt

loom:
	@$(WT) build-loom

test:
	@$(WT) test

clean:
	@$(WT) clean

distclean:
	@$(WT) distclean
