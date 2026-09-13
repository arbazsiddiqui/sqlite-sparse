/** Absolute path of the prebuilt `sparse0` extension for this platform. Throws on an unsupported one. */
export function getLoadablePath(): string;

/**
 * Loads `sparse0` into a database handle and returns it. Works with `node:sqlite`'s
 * `DatabaseSync` (opened with `allowExtension: true`), better-sqlite3 and node-sqlite3.
 */
export function load<T extends { loadExtension(path: string, ...rest: any[]): unknown }>(db: T): T;
