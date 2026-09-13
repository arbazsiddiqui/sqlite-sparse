"use strict";
const fs = require("node:fs");
const path = require("node:path");

// The filename is load-bearing: SQLite derives the entry point sqlite3_sparse_init from it.
const BINARIES = {
  "darwin-arm64": "sparse0.dylib",
  "linux-x64": "sparse0.so",
};

function getLoadablePath() {
  const platform = `${process.platform}-${process.arch}`;
  const file = BINARIES[platform];
  if (!file) {
    throw new Error(
      `sqlite-sparse: no prebuilt sparse0 for ${platform}. Build it from source ` +
        "(https://github.com/arbazsiddiqui/sqlite-sparse#development) and pass the binary to loadExtension().",
    );
  }
  const p = path.join(__dirname, "lib", platform, file);
  if (!fs.existsSync(p)) {
    throw new Error(`sqlite-sparse: ${p} is missing; the package was installed without its binaries.`);
  }
  return p;
}

// node:sqlite gates loadExtension behind enableLoadExtension (and allowExtension at open);
// better-sqlite3 and node-sqlite3 expose loadExtension directly.
function load(db) {
  const gated = typeof db.enableLoadExtension === "function";
  if (gated) db.enableLoadExtension(true);
  db.loadExtension(getLoadablePath());
  if (gated) db.enableLoadExtension(false);
  return db;
}

module.exports = { getLoadablePath, load };
