.PHONY: init build rebuild check clean purge

# Detect linked worktree: .git is a file with gitdir pointing to .git/worktrees/
IS_LINKED_WORKTREE := $(shell [ -f .git ] && grep -q '/.git/worktrees/' .git && echo 1 || echo 0)

init:
ifeq ($(IS_LINKED_WORKTREE),1)
	@# Derive main worktree root from .git file (same logic as CMakeLists.txt)
	@GITDIR_RAW=$$(sed 's/gitdir: //' .git | tr -d '\n\r'); \
	GITDIR=$$(cd "$$GITDIR_RAW" 2>/dev/null && pwd); \
	MAIN_ROOT=$$(dirname "$$(dirname "$$(dirname "$$GITDIR")")"); \
	echo "Linked worktree detected. Main worktree: $$MAIN_ROOT"; \
	\
	if [ ! -f "$$MAIN_ROOT/build/RISCV/gem5.opt" ]; then \
	  echo "ERROR: main worktree has not built gem5."; \
	  echo "  Expected: $$MAIN_ROOT/build/RISCV/gem5.opt"; \
	  echo "  Run gem5 build in $$MAIN_ROOT first."; \
	  exit 1; \
	fi; \
	\
	mkdir -p build; \
	if [ -d build/RISCV ] && [ ! -L build/RISCV ]; then \
	  echo "  build/RISCV is a real directory (local gem5 build exists), keeping it."; \
	else \
	  ln -sfn "$$MAIN_ROOT/build/RISCV" build/RISCV; \
	  echo "  Symlinked build/RISCV -> $$MAIN_ROOT/build/RISCV"; \
	fi
	git submodule update --init --depth 1 --filter=blob:none externals/gem5
	git submodule update --init --depth 1 --filter=blob:none externals/or-tools
else
	git submodule update --init --depth 1 --filter=blob:none externals/llvm
	git submodule update --init --depth 1 --filter=blob:none externals/circt
	git submodule update --init --depth 1 --filter=blob:none externals/gem5
	git submodule update --init --depth 1 --filter=blob:none externals/or-tools
endif

build:
	cmake -G Ninja -B build -S . -DCMAKE_BUILD_TYPE=Release $(CMAKE_EXTRA_ARGS)
	ninja -C build loom

rebuild:
	cmake -G Ninja -B build -S . -DCMAKE_BUILD_TYPE=Release $(CMAKE_EXTRA_ARGS)
	ninja -C build

check:
	ninja -C build check-loom

clean:
	ninja -C build -t clean 2>/dev/null || true
	rm -rf out node_modules
	rm -f package-lock.json package.json screenshot.mjs

purge:
	rm -rf build .cache
