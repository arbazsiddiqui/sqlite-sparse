# Needs cmake, a C/C++ toolchain and the SQLite headers.
SQLITE_INC ?= $(shell [ "$$(uname)" = Darwin ] && echo $$(brew --prefix sqlite)/include || echo /usr/include)

all:
	cmake -B build -DSQLITE_INCLUDE_DIR=$(SQLITE_INC) -DGGML_METAL=OFF -DGGML_BLAS=OFF
	cmake --build build -j

test: all
	python3 -m pip install -q -e "bindings/python[dev]"
	python3 -m pytest -q

clean:
	rm -rf build

.PHONY: all test clean
