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
#                    the same lock. This builds the configured CIRCT
#                    package's default target; it is not narrowed to the
#                    libraries loom links against, and CIRCT's tools are
#                    left enabled. The shared CIRCT build is owned by the
#                    main worktree's externals; main and linked invocations
#                    alike route to those shared outputs.
#   make or-tools  - build and install the exact shared OR-Tools package.
#   make loom      - build this worktree's loom build (auto-builds LLVM
#                    and OR-Tools when missing or when their build identities
#                    drifted;
#                    never builds CIRCT, but offers an already-built
#                    shared CIRCT via -DCIRCT_DIR when one matches)
#   make test      - run the complete Loom lit test suite (target: check-loom)
#   make sync-worktree - preflight and synchronize a linked branch with main
#   make clean     - remove this worktree's loom build only
#   make distclean - main worktree: remove the loom build and shared LLVM,
#                    CIRCT, and OR-Tools builds. Linked worktree: remove only
#                    this loom build (shared builds are left alone).

ROOT          := $(abspath $(CURDIR))
PYTHON        ?= python3
JOBS          ?= $(shell nproc)
LOCK_TIMEOUT  ?= 1800

WT_SCRIPT     := $(ROOT)/scripts/make-worktree.py
SYNC_SCRIPT   := $(ROOT)/scripts/sync_branches.py
WT            := $(PYTHON) $(WT_SCRIPT) \
                   --root $(ROOT) \
                   --jobs $(JOBS) \
                   --lock-timeout $(LOCK_TIMEOUT)

# LIT_OPTS is consulted by `make test`; export it so the dispatcher's
# environment matches an interactive shell invocation.
export LIT_OPTS
export JOBS

.PHONY: all doctor llvm circt or-tools loom test sync-worktree clean distclean

all: loom

doctor:
	@$(WT) doctor

llvm:
	@$(WT) build-llvm

circt:
	@$(WT) build-circt

or-tools:
	@$(WT) build-or-tools

loom:
	@$(WT) build-loom

test:
	@$(WT) test

sync-worktree:
	@git_dir="$$(git rev-parse --path-format=absolute --git-dir)"; \
	 common_dir="$$(git rev-parse --path-format=absolute --git-common-dir)"; \
	 if [ "$$git_dir" = "$$common_dir" ]; then \
	   echo "sync-worktree requires a linked worktree" >&2; \
	   exit 2; \
	 fi
	@$(PYTHON) $(SYNC_SCRIPT) main --dry-run
	@$(PYTHON) $(SYNC_SCRIPT) main

clean:
	@$(WT) clean

distclean:
	@$(WT) distclean
