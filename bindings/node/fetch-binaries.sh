#!/bin/sh
# Puts the GitHub release binaries for this package's version under lib/<platform>/.
# npm runs it before pack and publish; run it by hand after a clone.
set -eu
cd "$(dirname "$0")"
V=$(node -p "require('./package.json').version")
BASE="https://github.com/arbazsiddiqui/sqlite-sparse/releases/download/v$V"

[ "$(cat lib/VERSION 2>/dev/null)" = "$V" ] || rm -rf lib
fetch() { # <node platform> <release platform> <binary>
  mkdir -p "lib/$1"
  [ -s "lib/$1/$3" ] && return
  curl -fsSL "$BASE/sparse0-$V-loadable-$2.tar.gz" | tar xz -C "lib/$1" "$3"
  echo "lib/$1/$3"
}
fetch linux-x64 linux-x86_64 sparse0.so
fetch darwin-arm64 macos-arm64 sparse0.dylib
echo "$V" > lib/VERSION
