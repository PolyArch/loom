.PHONY: init build rebuild check clean purge

# Detect linked worktree: .git is a file with gitdir pointing to .git/worktrees/
IS_LINKED_WORKTREE := $(shell [ -f .git ] && grep -q '/.git/worktrees/' .git && echo 1 || echo 0)

init:
ifeq ($(IS_LINKED_WORKTREE),1)
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
