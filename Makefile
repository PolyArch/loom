.PHONY: init build rebuild check clean purge

init:
	git submodule update --init --depth 1 --filter=blob:none externals/llvm
	git submodule update --init --depth 1 --filter=blob:none externals/circt
	git submodule update --init --depth 1 --filter=blob:none externals/gem5
	git submodule update --init --depth 1 --filter=blob:none externals/or-tools

build:
	cmake -G Ninja -B build -S . -DCMAKE_BUILD_TYPE=Release
	ninja -C build fcc

rebuild:
	cmake -G Ninja -B build -S . -DCMAKE_BUILD_TYPE=Release
	ninja -C build

check:
	ninja -C build check-fcc

clean:
	ninja -C build -t clean 2>/dev/null || true
	rm -rf out node_modules
	rm -f package-lock.json package.json screenshot.mjs

purge:
	rm -rf build .cache
