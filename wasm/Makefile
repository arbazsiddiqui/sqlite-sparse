# Query-only sparse0 compiled into the official SQLite WebAssembly build.
# Needs emcc (Emscripten), wabt, tclsh and curl. Output: dist/sqlite3.mjs, dist/sqlite3.wasm.
#
#   make -C wasm            # build
#   make -C wasm check      # run the Node smoke test against dist/
#
SQLITE_VERSION ?= 3530400
SQLITE_YEAR    ?= 2026
UTF8PROC_REF   ?= e5e799221b45bbb90f5fdc5c69b6b8dfbf017e78
BUILD          := build
SRC            := $(BUILD)/sqlite-src-$(SQLITE_VERSION)
UTF8PROC       := $(BUILD)/utf8proc
LIB            := $(BUILD)/libsparse0.wasm.a
DIST           := dist
# SQLite's WASM makefile is written for GNU sed; on macOS `brew install gnu-sed` and its gnubin
# directory goes first on the PATH for that step.
GNUBIN         := $(wildcard /opt/homebrew/opt/gnu-sed/libexec/gnubin /usr/local/opt/gnu-sed/libexec/gnubin)
WASM_OPT       := $(shell command -v wasm-opt || ls /opt/homebrew/opt/emscripten/libexec/binaryen/bin/wasm-opt 2>/dev/null || true)

all: $(DIST)/sqlite3.mjs $(DIST)/sqlite3.wasm

$(BUILD)/sqlite-src.zip:
	mkdir -p $(BUILD)
	curl -fsSL -o $@ https://sqlite.org/$(SQLITE_YEAR)/sqlite-src-$(SQLITE_VERSION).zip

$(SRC)/sqlite3.c: $(BUILD)/sqlite-src.zip
	cd $(BUILD) && rm -rf sqlite-src-$(SQLITE_VERSION) && unzip -q sqlite-src.zip
	cd $(SRC) && ./configure --enable-all >/dev/null && $(MAKE) sqlite3.c >/dev/null

$(UTF8PROC)/utf8proc.c:
	mkdir -p $(BUILD)
	rm -rf $(UTF8PROC) && git clone -q https://github.com/JuliaStrings/utf8proc $(UTF8PROC) && cd $(UTF8PROC) && git checkout -q $(UTF8PROC_REF)

$(LIB): sqlite3_wasm_extra_init.c ../src/*.c $(SRC)/sqlite3.c $(UTF8PROC)/utf8proc.c
	emcc -O3 -DSQLITE_CORE -DSPARSE0_QUERY_ONLY -DUTF8PROC_STATIC -I$(SRC) -I$(UTF8PROC) -I../src \
	  -Wall -Wno-unused-function -c sqlite3_wasm_extra_init.c -o $(BUILD)/sparse0_wasm.o
	emar rcs $@ $(BUILD)/sparse0_wasm.o

# SQLite's configure only writes ext/wasm/config.make when it finds an EMSDK tree; a
# Homebrew or distro Emscripten has emcc on the PATH instead, so write it directly.
$(SRC)/ext/wasm/config.make: $(SRC)/sqlite3.c
	@command -v emcc >/dev/null || { echo "emcc not found"; exit 1; }
	@command -v wasm-strip >/dev/null || { echo "wasm-strip (wabt) not found"; exit 1; }
	printf 'bin.bash = %s\nbin.emcc = %s\nbin.wasm-strip = %s\nbin.wasm-opt = %s\nSHELL = %s\n' \
	  "$$(command -v bash)" "$$(command -v emcc)" "$$(command -v wasm-strip)" "$(WASM_OPT)" "$$(command -v bash)" > $@

$(DIST)/sqlite3.mjs $(DIST)/sqlite3.wasm: $(LIB) $(SRC)/ext/wasm/config.make
	cd $(SRC)/ext/wasm && PATH="$(GNUBIN):$$PATH" $(MAKE) sqlite3_wasm_extra_init.c=$(abspath $(LIB)) jswasm/sqlite3.mjs jswasm/sqlite3.wasm
	mkdir -p $(DIST)
	@out=$(SRC)/ext/wasm/jswasm; [ -f $$out/esm/sqlite3.mjs ] && out=$$out/esm; cp $$out/sqlite3.mjs $$out/sqlite3.wasm $(DIST)/
	ls -la $(DIST)

# The ESM bundle is browser-only, so the check runs in a browser: serve this directory and
# open check.html, which loads dist/ and prints its results.
check: all
	@echo "open http://localhost:8767/check.html  (Ctrl-C to stop)"
	python3 -m http.server 8767

clean:
	rm -rf $(BUILD) $(DIST)

.PHONY: all check clean
