/* sparse0: SQLite virtual table for learned sparse retrieval.
 * File format sqlite-sparse/1, shared with the Python reference implementation. */
#include "sqlite3ext.h"
SQLITE_EXTENSION_INIT1

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sha256.c"
#include "wordpiece.c"
#include "head.c"
#include "scorer.c"
#include "encoder.c"

#define MAX_QUERY_TERMS 512
#define MAX_DOC_TERMS 4096
#define DEFAULT_K 10
#define WEIGHT_SCALE_DEFAULT 40.0f

/* model registry, process-wide */

typedef struct {
    char *name;
    char *gguf_path;
    SparseHead *head;        /* loaded eagerly at register */
    SparseEncoder *enc;      /* loaded lazily on first INSERT */
    int n_threads;
    int max_seq;             /* encoder truncation; 512 = the models' card setting */
    char encoder_sha[65];    /* sha256 of the GGUF file */
    char sidecar_sha[65];    /* sha256 of the .sprs file */
    char vocab_sha[65];      /* sha256 of the newline-joined vocabulary */
} SparseModel;

#define MAX_MODELS 16
static SparseModel g_models[MAX_MODELS];
static int g_n_models = 0;

static SparseModel *model_find(const char *name) {
    for (int i = 0; i < g_n_models; i++)
        if (strcmp(g_models[i].name, name) == 0) return &g_models[i];
    return NULL;
}

/* The vocabulary is stored in meta as a JSON array of strings. */

static void utf8_push(char **w, uint32_t cp) {
    char *p = *w;
    if (cp < 0x80) *p++ = (char)cp;
    else if (cp < 0x800) { *p++ = 0xC0 | (cp >> 6); *p++ = 0x80 | (cp & 0x3F); }
    else if (cp < 0x10000) { *p++ = 0xE0 | (cp >> 12); *p++ = 0x80 | ((cp >> 6) & 0x3F); *p++ = 0x80 | (cp & 0x3F); }
    else { *p++ = 0xF0 | (cp >> 18); *p++ = 0x80 | ((cp >> 12) & 0x3F);
           *p++ = 0x80 | ((cp >> 6) & 0x3F); *p++ = 0x80 | (cp & 0x3F); }
    *w = p;
}

static int hex4(const char *s) {
    int v = 0;
    for (int i = 0; i < 4; i++) {
        char c = s[i]; v <<= 4;
        if (c >= '0' && c <= '9') v |= c - '0';
        else if (c >= 'a' && c <= 'f') v |= c - 'a' + 10;
        else if (c >= 'A' && c <= 'F') v |= c - 'A' + 10;
        else return -1;
    }
    return v;
}

/* JSON array of strings -> newline-joined blob. NULL on error. */
static char *json_vocab_to_blob(const char *js, int *n_out) {
    size_t cap = strlen(js) + 1;
    char *blob = malloc(cap), *w = blob;
    const char *p = js;
    int n = 0;
    while (*p && *p != '[') p++;
    if (*p != '[') { free(blob); return NULL; }
    p++;
    for (;;) {
        while (*p == ' ' || *p == ',' || *p == '\n' || *p == '\t' || *p == '\r') p++;
        if (*p == ']') break;
        if (*p != '"') { free(blob); return NULL; }
        p++;
        if (n++) *w++ = '\n';
        while (*p && *p != '"') {
            if (*p == '\\') {
                p++;
                switch (*p) {
                    case '"': *w++ = '"'; p++; break;
                    case '\\': *w++ = '\\'; p++; break;
                    case '/': *w++ = '/'; p++; break;
                    case 'b': *w++ = '\b'; p++; break;
                    case 'f': *w++ = '\f'; p++; break;
                    case 'n': *w++ = '\n'; p++; break;
                    case 'r': *w++ = '\r'; p++; break;
                    case 't': *w++ = '\t'; p++; break;
                    case 'u': {
                        int u = hex4(p + 1);
                        if (u < 0) { free(blob); return NULL; }
                        p += 5;
                        if (u >= 0xD800 && u <= 0xDBFF && p[0] == '\\' && p[1] == 'u') {
                            int lo = hex4(p + 2);
                            if (lo >= 0xDC00 && lo <= 0xDFFF) {
                                utf8_push(&w, 0x10000 + (((uint32_t)(u - 0xD800)) << 10) + (lo - 0xDC00));
                                p += 6;
                                break;
                            }
                        }
                        utf8_push(&w, (uint32_t)u);
                        break;
                    }
                    default: free(blob); return NULL;
                }
            } else *w++ = *p++;
        }
        if (*p != '"') { free(blob); return NULL; }
        p++;
    }
    *w = 0;
    if (n_out) *n_out = n;
    return blob;
}

/* newline blob -> JSON array */
static char *blob_to_json_vocab(const char *blob) {
    size_t cap = strlen(blob) * 6 + 16;
    char *js = malloc(cap), *w = js;
    *w++ = '[';
    const char *p = blob;
    int first = 1;
    while (*p) {
        const char *e = strchr(p, '\n');
        size_t len = e ? (size_t)(e - p) : strlen(p);
        if (!first) *w++ = ',';
        first = 0;
        *w++ = '"';
        for (size_t i = 0; i < len; i++) {
            unsigned char c = p[i];
            if (c == '"' || c == '\\') { *w++ = '\\'; *w++ = c; }
            else if (c < 0x20) { w += sprintf(w, "\\u%04x", c); }
            else *w++ = c;
        }
        *w++ = '"';
        if (!e) break;
        p = e + 1;
    }
    *w++ = ']'; *w = 0;
    return js;
}

/* virtual table */

typedef struct {
    sqlite3_vtab base;
    sqlite3 *db;
    char *model_name;        /* registry key from CREATE args */
    /* query-side state, read from the database file */
    Vocab *vocab;
    float *qlut;             /* [vocab_n] */
    int vocab_n;
    float weight_scale;
    int weight_mode_u8;
    int loaded;
    /* refreshed when PRAGMA data_version changes */
    int ndocs;
    int64_t *dead;
    int n_dead;
    int64_t data_version;
    ScoreCtx score;
    int model_checked;       /* registered model verified against this file */
} Sparse0Tab;

static int meta_get_text(sqlite3 *db, const char *k, char **out);

/* older files lack the ntokens/truncated columns */
static void docs_upgrade_columns(sqlite3 *db) {
    int has = 0;
    sqlite3_stmt *st;
    if (sqlite3_prepare_v2(db, "PRAGMA table_info(docs)", -1, &st, NULL) == SQLITE_OK) {
        while (sqlite3_step(st) == SQLITE_ROW)
            if (strcmp((const char *)sqlite3_column_text(st, 1), "ntokens") == 0) has = 1;
        sqlite3_finalize(st);
    }
    if (!has) {
        sqlite3_exec(db, "ALTER TABLE docs ADD COLUMN ntokens INTEGER", NULL, NULL, NULL);
        sqlite3_exec(db, "ALTER TABLE docs ADD COLUMN truncated INTEGER", NULL, NULL, NULL);
    }
}

/* Postings are only valid under the vocabulary and query table they were
 * built with; refuse INSERT from a model whose tables differ. */
static int model_matches_file(sqlite3 *db, SparseModel *m, char **why) {
    char *vjson = NULL;
    meta_get_text(db, "vocab", &vjson);
    if (!vjson) { *why = sqlite3_mprintf("file has no vocabulary"); return 0; }
    int n = 0;
    char *blob = json_vocab_to_blob(vjson, &n);
    sqlite3_free(vjson);
    if (!blob) { *why = sqlite3_mprintf("file vocabulary unreadable"); return 0; }
    char sha[65];
    sha256_bytes_hex(blob, strlen(blob), sha);
    free(blob);
    if (strcmp(sha, m->vocab_sha) != 0) {
        *why = sqlite3_mprintf("vocabulary differs (file %.12s..., model '%s' %.12s...)", sha, m->name, m->vocab_sha);
        return 0;
    }
    sqlite3_stmt *st;
    if (sqlite3_prepare_v2(db, "SELECT t, w FROM qlut", -1, &st, NULL) != SQLITE_OK) { *why = sqlite3_mprintf("no query table"); return 0; }
    int rows = 0, bad = 0;
    while (sqlite3_step(st) == SQLITE_ROW) {
        int t = sqlite3_column_int(st, 0);
        double w = sqlite3_column_double(st, 1);
        rows++;
        if (t < 0 || (uint32_t)t >= m->head->vocab_n || fabs(w - (double)m->head->qlut[t]) > 1e-5) bad++;
    }
    sqlite3_finalize(st);
    if (bad || rows == 0) {
        *why = sqlite3_mprintf("query weight table differs from model '%s' (%d of %d entries)", m->name, bad, rows);
        return 0;
    }
    return 1;
}

typedef struct {
    sqlite3_vtab_cursor base;
    Hit *hits;
    int n_hits;
    int pos;
} Sparse0Cur;

static int meta_get_text(sqlite3 *db, const char *k, char **out) {
    sqlite3_stmt *st;
    int rc = sqlite3_prepare_v2(db, "SELECT v FROM meta WHERE k=?", -1, &st, NULL);
    if (rc != SQLITE_OK) return rc;
    sqlite3_bind_text(st, 1, k, -1, SQLITE_STATIC);
    *out = NULL;
    if (sqlite3_step(st) == SQLITE_ROW)
        *out = sqlite3_mprintf("%s", sqlite3_column_text(st, 0));
    sqlite3_finalize(st);
    return SQLITE_OK;
}

static int tab_load_query_state(Sparse0Tab *t) {
    if (t->loaded) return SQLITE_OK;
    char *fmt = NULL, *vjson = NULL, *scale = NULL, *mode = NULL;
    meta_get_text(t->db, "format", &fmt);
    if (!fmt || strcmp(fmt, "sqlite-sparse/1") != 0) { sqlite3_free(fmt); return SQLITE_ERROR; }
    sqlite3_free(fmt);
    meta_get_text(t->db, "vocab", &vjson);
    if (!vjson) return SQLITE_ERROR;
    int n = 0;
    char *blob = json_vocab_to_blob(vjson, &n);
    sqlite3_free(vjson);
    if (!blob) return SQLITE_ERROR;
    t->vocab = vocab_from_blob(blob);
    t->vocab_n = n;
    meta_get_text(t->db, "weight_scale", &scale);
    t->weight_scale = scale ? (float)atof(scale) : WEIGHT_SCALE_DEFAULT;
    sqlite3_free(scale);
    meta_get_text(t->db, "weight_mode", &mode);
    t->weight_mode_u8 = !mode || strcmp(mode, "u8") == 0;
    sqlite3_free(mode);
    t->qlut = calloc(n, sizeof(float));
    sqlite3_stmt *st;
    if (sqlite3_prepare_v2(t->db, "SELECT t, w FROM qlut", -1, &st, NULL) != SQLITE_OK) return SQLITE_ERROR;
    while (sqlite3_step(st) == SQLITE_ROW) {
        int term = sqlite3_column_int(st, 0);
        if (term >= 0 && term < n) t->qlut[term] = (float)sqlite3_column_double(st, 1);
    }
    sqlite3_finalize(st);
    t->loaded = 1;
    return SQLITE_OK;
}

static int64_t db_data_version(sqlite3 *db) {
    sqlite3_stmt *st;
    int64_t v = -1;
    if (sqlite3_prepare_v2(db, "PRAGMA data_version", -1, &st, NULL) == SQLITE_OK) {
        if (sqlite3_step(st) == SQLITE_ROW) v = sqlite3_column_int64(st, 0);
        sqlite3_finalize(st);
    }
    return v;
}

/* data_version is bumped by any commit from any connection */
static void tab_refresh_doc_state(Sparse0Tab *t) {
    int64_t dv = db_data_version(t->db);
    if (dv == t->data_version && t->ndocs > 0) return;
    t->data_version = dv;
    char *nd = NULL;
    meta_get_text(t->db, "ndocs", &nd);
    t->ndocs = nd ? atoi(nd) : 0;
    sqlite3_free(nd);
    free(t->dead);
    t->dead = NULL;
    t->n_dead = 0;
    int cap = 0;
    sqlite3_stmt *st;
    if (sqlite3_prepare_v2(t->db, "SELECT id FROM docs WHERE deleted=1", -1, &st, NULL) == SQLITE_OK) {
        while (sqlite3_step(st) == SQLITE_ROW) {
            if (t->n_dead == cap) { cap = cap ? cap * 2 : 64; t->dead = realloc(t->dead, cap * sizeof(int64_t)); }
            t->dead[t->n_dead++] = sqlite3_column_int64(st, 0);
        }
        sqlite3_finalize(st);
    }
}

static const char *SCHEMA_SQL =
    "CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT);"
    "CREATE TABLE IF NOT EXISTS qlut(t INTEGER PRIMARY KEY, w REAL);"
    "CREATE TABLE IF NOT EXISTS postings(t INTEGER PRIMARY KEY, docs BLOB, ws BLOB);"
    "CREATE TABLE IF NOT EXISTS docs(id INTEGER PRIMARY KEY AUTOINCREMENT,"
    " ext_id TEXT UNIQUE, title TEXT, body TEXT, meta TEXT, deleted INTEGER DEFAULT 0,"
    " ntokens INTEGER, truncated INTEGER);"
    "CREATE TABLE IF NOT EXISTS pending(id INTEGER PRIMARY KEY AUTOINCREMENT,"
    " ext_id TEXT, title TEXT, body TEXT NOT NULL);"
    "CREATE INDEX IF NOT EXISTS docs_deleted ON docs(deleted) WHERE deleted=1;";

static int sparse0_init_model_rows(sqlite3 *db, SparseModel *m) {
    char *fmt = NULL;
    meta_get_text(db, "format", &fmt);
    if (fmt) { sqlite3_free(fmt); return SQLITE_OK; }   /* already initialized */
    char *json = blob_to_json_vocab(m->head->vocab_blob);
    char *sql = sqlite3_mprintf(
        "INSERT OR REPLACE INTO meta VALUES"
        "('format','sqlite-sparse/1'),('model_id',%Q),"
        "('weight_mode','u8'),('weight_scale','40.0'),('ndocs','0'),"
        "('encoder_sha256',%Q),('sidecar_sha256',%Q),('vocab_sha256',%Q),"
        "('max_seq',%d),"
        "('vocab',%Q);", m->name, m->encoder_sha, m->sidecar_sha, m->vocab_sha, m->max_seq, json);
    int rc = sqlite3_exec(db, sql, NULL, NULL, NULL);
    sqlite3_free(sql);
    free(json);
    if (rc != SQLITE_OK) return rc;
    sqlite3_stmt *st;
    rc = sqlite3_prepare_v2(db, "INSERT OR REPLACE INTO qlut VALUES(?,?)", -1, &st, NULL);
    if (rc != SQLITE_OK) return rc;
    for (uint32_t v = 0; v < m->head->vocab_n; v++) {
        if (m->head->qlut[v] == 0.0f) continue;
        sqlite3_bind_int(st, 1, (int)v);
        sqlite3_bind_double(st, 2, (double)m->head->qlut[v]);
        sqlite3_step(st);
        sqlite3_reset(st);
    }
    sqlite3_finalize(st);
    return SQLITE_OK;
}

static int sparse0_connect_common(sqlite3 *db, void *aux, int argc,
                                  const char *const *argv, sqlite3_vtab **out,
                                  char **err, int create) {
    (void)aux;
    const char *model = NULL;
    for (int i = 3; i < argc; i++) {
        if (strncmp(argv[i], "model", 5) == 0) {
            const char *eq = strchr(argv[i], '=');
            if (eq) {
                eq++;
                while (*eq == ' ' || *eq == '\'' || *eq == '"') eq++;
                size_t len = strlen(eq);
                while (len && (eq[len-1] == ' ' || eq[len-1] == '\'' || eq[len-1] == '"')) len--;
                model = sqlite3_mprintf("%.*s", (int)len, eq);
            }
        }
    }
    if (create) {
        char *fmt = NULL;
        meta_get_text(db, "format", &fmt);
        int have_format = fmt && strcmp(fmt, "sqlite-sparse/1") == 0;
        sqlite3_free(fmt);
        if (!have_format) {

            if (!model) { *err = sqlite3_mprintf("sparse0: model='name' argument required for a new database"); return SQLITE_ERROR; }
            SparseModel *m = model_find(model);
            if (!m) { *err = sqlite3_mprintf("sparse0: model '%s' not registered (call sparse_register first)", model); return SQLITE_ERROR; }
            int rc = sqlite3_exec(db, SCHEMA_SQL, NULL, NULL, NULL);
            if (rc == SQLITE_OK) rc = sparse0_init_model_rows(db, m);
            if (rc != SQLITE_OK) { *err = sqlite3_mprintf("sparse0: init failed (%s)", sqlite3_errmsg(db)); return rc; }
        }

    }
    {
        char *fmt = NULL;
        meta_get_text(db, "format", &fmt);
        if (fmt) docs_upgrade_columns(db);
        sqlite3_free(fmt);
    }
    char *decl = sqlite3_mprintf(
        "CREATE TABLE x(text TEXT, score REAL, k INTEGER HIDDEN, \"%w\" TEXT HIDDEN)", argv[2]);
    int rc = sqlite3_declare_vtab(db, decl);
    sqlite3_free(decl);
    if (rc != SQLITE_OK) return rc;
    Sparse0Tab *t = sqlite3_malloc(sizeof(Sparse0Tab));
    memset(t, 0, sizeof(*t));
    t->db = db;
    t->model_name = model ? sqlite3_mprintf("%s", model) : NULL;
    *out = &t->base;
    return SQLITE_OK;
}

static int sparse0_create(sqlite3 *db, void *aux, int argc, const char *const *argv,
                          sqlite3_vtab **out, char **err) {
    return sparse0_connect_common(db, aux, argc, argv, out, err, 1);
}
static int sparse0_connect(sqlite3 *db, void *aux, int argc, const char *const *argv,
                           sqlite3_vtab **out, char **err) {
    return sparse0_connect_common(db, aux, argc, argv, out, err, 0);
}

static int sparse0_disconnect(sqlite3_vtab *vt) {
    Sparse0Tab *t = (Sparse0Tab *)vt;
    vocab_free(t->vocab);
    free(t->qlut);
    free(t->dead);
    score_ctx_free(&t->score);
    sqlite3_free(t->model_name);
    sqlite3_free(t);
    return SQLITE_OK;
}

/* idxNum bits: 1 MATCH, 2 k, 4 rowid lookup, 8 LIMIT */
static int sparse0_bestindex(sqlite3_vtab *vt, sqlite3_index_info *ii) {
    (void)vt;
    int i_match = -1, i_k = -1;
    for (int i = 0; i < ii->nConstraint; i++) {
        const struct sqlite3_index_constraint *c = &ii->aConstraint[i];
        if (!c->usable) continue;
        if (c->op == SQLITE_INDEX_CONSTRAINT_MATCH && (c->iColumn == 0 || c->iColumn == 3))
            i_match = i;
        if (c->op == SQLITE_INDEX_CONSTRAINT_EQ && c->iColumn == 2)
            i_k = i;
    }
    int i_rowid = -1;
    for (int i = 0; i < ii->nConstraint; i++) {
        const struct sqlite3_index_constraint *c = &ii->aConstraint[i];
        if (c->usable && c->op == SQLITE_INDEX_CONSTRAINT_EQ && c->iColumn == -1)
            i_rowid = i;
    }
    int i_limit = -1;
#ifdef SQLITE_INDEX_CONSTRAINT_LIMIT
    for (int i = 0; i < ii->nConstraint; i++)
        if (ii->aConstraint[i].usable && ii->aConstraint[i].op == SQLITE_INDEX_CONSTRAINT_LIMIT) i_limit = i;
#endif
    if (i_match >= 0) {
        ii->aConstraintUsage[i_match].argvIndex = 1;
        ii->aConstraintUsage[i_match].omit = 1;
        ii->idxNum = 1;
        int next = 2;
        if (i_k >= 0) {
            ii->aConstraintUsage[i_k].argvIndex = next++;
            ii->aConstraintUsage[i_k].omit = 1;
            ii->idxNum |= 2;
        }
        if (i_limit >= 0) {
            /* LIMIT feeds k when k is absent; SQLite still enforces it */
            ii->aConstraintUsage[i_limit].argvIndex = next++;
            ii->idxNum |= 8;
        }
        /* output is already ordered by score DESC, rowid ASC */
        if (ii->nOrderBy >= 1 && ii->nOrderBy <= 2
            && ii->aOrderBy[0].iColumn == 1 && ii->aOrderBy[0].desc
            && (ii->nOrderBy == 1 || (ii->aOrderBy[1].iColumn == -1 && !ii->aOrderBy[1].desc)))
            ii->orderByConsumed = 1;
        ii->estimatedCost = 10.0;
        ii->estimatedRows = DEFAULT_K;
    } else if (i_rowid >= 0) {

        ii->aConstraintUsage[i_rowid].argvIndex = 1;
        ii->aConstraintUsage[i_rowid].omit = 1;
        ii->idxNum = 4;
        ii->estimatedCost = 1.0;
        ii->estimatedRows = 1;
    } else {

        ii->idxNum = 0;
        ii->estimatedCost = 1e6;
        ii->estimatedRows = 100000;
    }
    return SQLITE_OK;
}

static int sparse0_open(sqlite3_vtab *vt, sqlite3_vtab_cursor **out) {
    (void)vt;
    Sparse0Cur *c = sqlite3_malloc(sizeof(Sparse0Cur));
    memset(c, 0, sizeof(*c));
    *out = &c->base;
    return SQLITE_OK;
}

static int sparse0_close(sqlite3_vtab_cursor *cur) {
    Sparse0Cur *c = (Sparse0Cur *)cur;
    free(c->hits);
    sqlite3_free(c);
    return SQLITE_OK;
}

static int sparse0_filter(sqlite3_vtab_cursor *cur, int idxNum, const char *idxStr,
                          int argc, sqlite3_value **argv) {
    (void)idxStr;
    Sparse0Cur *c = (Sparse0Cur *)cur;
    Sparse0Tab *t = (Sparse0Tab *)cur->pVtab;
    free(c->hits);
    c->hits = NULL; c->n_hits = 0; c->pos = 0;
    if (idxNum == 4 && argc >= 1) {          /* rowid point lookup */
        sqlite3_int64 rid = sqlite3_value_int64(argv[0]);
        sqlite3_stmt *st;
        if (sqlite3_prepare_v2(t->db, "SELECT id FROM docs WHERE id=? AND deleted=0", -1, &st, NULL) != SQLITE_OK)
            return SQLITE_ERROR;
        sqlite3_bind_int64(st, 1, rid);
        if (sqlite3_step(st) == SQLITE_ROW) {
            c->hits = malloc(sizeof(Hit));
            c->hits[0].doc = (int32_t)rid; c->hits[0].score = 0.0f;
            c->n_hits = 1;
        }
        sqlite3_finalize(st);
        return SQLITE_OK;
    }
    if (!(idxNum & 1)) {                     /* full scan of live docs */
        sqlite3_stmt *st;
        if (sqlite3_prepare_v2(t->db, "SELECT id FROM docs WHERE deleted=0 ORDER BY id", -1, &st, NULL) != SQLITE_OK)
            return SQLITE_ERROR;
        int cap = 1024;
        c->hits = malloc(cap * sizeof(Hit));
        while (sqlite3_step(st) == SQLITE_ROW) {
            if (c->n_hits == cap) { cap *= 2; c->hits = realloc(c->hits, cap * sizeof(Hit)); }
            c->hits[c->n_hits].doc = (int32_t)sqlite3_column_int64(st, 0);
            c->hits[c->n_hits].score = 0.0f;
            c->n_hits++;
        }
        sqlite3_finalize(st);
        return SQLITE_OK;
    }
    if (argc < 1) return SQLITE_OK;
    if (tab_load_query_state(t) != SQLITE_OK) {
        t->base.zErrMsg = sqlite3_mprintf("sparse0: not a sqlite-sparse/1 database");
        return SQLITE_ERROR;
    }
    const char *q = (const char *)sqlite3_value_text(argv[0]);
    int ai = 1, k = 0;
    if (idxNum & 2) k = sqlite3_value_int(argv[ai++]);
    if ((idxNum & 8) && k <= 0 && argc > ai) k = sqlite3_value_int(argv[ai]);
    if (k <= 0) k = DEFAULT_K;
    if (!q) return SQLITE_OK;

    int32_t ids[MAX_QUERY_TERMS];
    int n_ids = wp_tokenize(t->vocab, q, ids, MAX_QUERY_TERMS);

    int32_t qterms[MAX_QUERY_TERMS];
    float qweights[MAX_QUERY_TERMS];
    int nq = 0;
    for (int i = 0; i < n_ids; i++) {
        int32_t term = ids[i];
        if (term < 0 || term >= t->vocab_n) continue;
        float w = t->qlut[term];
        if (w == 0.0f) continue;
        int found = -1;
        for (int j = 0; j < nq; j++) if (qterms[j] == term) { found = j; break; }
        if (found >= 0) qweights[found] += w;
        else { qterms[nq] = term; qweights[nq] = w; nq++; }
    }
    if (!nq) return SQLITE_OK;

    tab_refresh_doc_state(t);
    int ndocs = t->ndocs;
    if (!ndocs) return SQLITE_OK;

    /* deleted docs are filtered after scoring */
    int n_dead = t->n_dead;
    const int64_t *dead = t->dead;
    int want = k + n_dead;
    Hit *hits = malloc((want > 0 ? want : 1) * sizeof(Hit));
    int n = sparse_score_ctx(&t->score, t->db, qterms, qweights, nq, ndocs,
                             t->weight_scale, t->weight_mode_u8, want, hits);
    if (n < 0) { free(hits); return SQLITE_ERROR; }

    int m = 0;
    for (int i = 0; i < n && m < k; i++) {
        int is_dead = 0;
        for (int j = 0; j < n_dead; j++) if (dead[j] == hits[i].doc) { is_dead = 1; break; }
        if (!is_dead) hits[m++] = hits[i];
    }
    c->hits = hits;
    c->n_hits = m;
    return SQLITE_OK;
}

static int sparse0_next(sqlite3_vtab_cursor *cur) { ((Sparse0Cur *)cur)->pos++; return SQLITE_OK; }
static int sparse0_eof(sqlite3_vtab_cursor *cur) {
    Sparse0Cur *c = (Sparse0Cur *)cur;
    return c->pos >= c->n_hits;
}
static int sparse0_column(sqlite3_vtab_cursor *cur, sqlite3_context *ctx, int col) {
    Sparse0Cur *c = (Sparse0Cur *)cur;
    if (col == 1) sqlite3_result_double(ctx, c->hits[c->pos].score);
    else sqlite3_result_null(ctx);
    return SQLITE_OK;
}
static int sparse0_rowid(sqlite3_vtab_cursor *cur, sqlite3_int64 *out) {
    Sparse0Cur *c = (Sparse0Cur *)cur;
    *out = c->hits[c->pos].doc;
    return SQLITE_OK;
}

/* write path */

static int sparse0_insert_doc(Sparse0Tab *t, sqlite3_int64 rowid_in,
                              const char *text, sqlite3_int64 *rowid_out) {
    SparseModel *m = t->model_name ? model_find(t->model_name) : NULL;
    if (!m) {
        t->base.zErrMsg = sqlite3_mprintf("sparse0: model '%s' not registered in this process",
                                          t->model_name ? t->model_name : "(none)");
        return SQLITE_ERROR;
    }
    if (!m->enc) {
        m->enc = sparse_encoder_load(m->gguf_path, m->n_threads, m->max_seq);
        if (!m->enc) {
            t->base.zErrMsg = sqlite3_mprintf("sparse0: cannot load encoder %s", m->gguf_path);
            return SQLITE_ERROR;
        }
    }
    if (!t->model_checked) {
        char *why = NULL;
        if (!model_matches_file(t->db, m, &why)) {
            t->base.zErrMsg = sqlite3_mprintf("sparse0: this index was built with a different model. %s", why);
            sqlite3_free(why);
            return SQLITE_ERROR;
        }
        t->model_checked = 1;
    }
    float *hs = NULL;
    int n_total = 0;
    int n_tok = sparse_encoder_encode(m->enc, text, &hs, &n_total);
    if (n_tok <= 0) {
        t->base.zErrMsg = sqlite3_mprintf("sparse0: encode failed");
        return SQLITE_ERROR;
    }
    int32_t terms[MAX_DOC_TERMS];
    float weights[MAX_DOC_TERMS];
    int n_terms = sparse_head_apply(m->head, hs, n_tok, 1.0f / (2.0f * WEIGHT_SCALE_DEFAULT),
                                    terms, weights, MAX_DOC_TERMS);
    free(hs);
    if (n_terms < 0) {
        t->base.zErrMsg = sqlite3_mprintf("sparse0: head failed");
        return SQLITE_ERROR;
    }

    sqlite3 *db = t->db;
    sqlite3_stmt *st;
    sqlite3_int64 did;
    int truncated = n_total > n_tok;
    if (rowid_in > 0) {
        sqlite3_prepare_v2(db, "INSERT INTO docs(id, ext_id, ntokens, truncated) VALUES(?, ?, ?, ?)", -1, &st, NULL);
        sqlite3_bind_int64(st, 1, rowid_in);
        char idbuf[32];
        snprintf(idbuf, sizeof(idbuf), "%lld", (long long)rowid_in);
        sqlite3_bind_text(st, 2, idbuf, -1, SQLITE_TRANSIENT);
        sqlite3_bind_int(st, 3, n_total);
        sqlite3_bind_int(st, 4, truncated);
        int rc = sqlite3_step(st);
        sqlite3_finalize(st);
        if (rc != SQLITE_DONE) {
            t->base.zErrMsg = sqlite3_mprintf("sparse0: rowid insert failed (%s)", sqlite3_errmsg(db));
            return SQLITE_ERROR;
        }
        did = rowid_in;
    } else {
        sqlite3_prepare_v2(db, "INSERT INTO docs(ext_id, ntokens, truncated) VALUES(NULL, ?, ?)", -1, &st, NULL);
        sqlite3_bind_int(st, 1, n_total);
        sqlite3_bind_int(st, 2, truncated);
        sqlite3_step(st);
        sqlite3_finalize(st);
        did = sqlite3_last_insert_rowid(db);
        sqlite3_stmt *fix;
        sqlite3_prepare_v2(db, "UPDATE docs SET ext_id=? WHERE id=?", -1, &fix, NULL);
        char idbuf[32];
        snprintf(idbuf, sizeof(idbuf), "%lld", (long long)did);
        sqlite3_bind_text(fix, 1, idbuf, -1, SQLITE_TRANSIENT);
        sqlite3_bind_int64(fix, 2, did);
        sqlite3_step(fix);
        sqlite3_finalize(fix);
    }

    /* weights quantized to u8 at scale 40, clamped to [1,255] */
    sqlite3_stmt *sel, *ins;
    sqlite3_prepare_v2(db, "SELECT docs, ws FROM postings WHERE t=?", -1, &sel, NULL);
    sqlite3_prepare_v2(db, "INSERT OR REPLACE INTO postings VALUES(?,?,?)", -1, &ins, NULL);
    for (int i = 0; i < n_terms; i++) {
        int32_t term = terms[i];
        float w = weights[i] * WEIGHT_SCALE_DEFAULT;
        uint8_t wq = w < 1.0f ? 1 : w > 255.0f ? 255 : (uint8_t)lrintf(w);
        sqlite3_bind_int(sel, 1, term);
        const void *odocs = NULL, *ows = NULL;
        int on = 0;
        if (sqlite3_step(sel) == SQLITE_ROW) {
            on = sqlite3_column_bytes(sel, 0) / 4;
            odocs = sqlite3_column_blob(sel, 0);
            ows = sqlite3_column_blob(sel, 1);
        }
        int32_t *nd = malloc((on + 1) * 4);
        uint8_t *nw = malloc(on + 1);
        if (on) { memcpy(nd, odocs, on * 4); memcpy(nw, ows, on); }
        nd[on] = (int32_t)did;
        nw[on] = wq;
        sqlite3_reset(sel);
        sqlite3_bind_int(ins, 1, term);
        sqlite3_bind_blob(ins, 2, nd, (on + 1) * 4, free);
        sqlite3_bind_blob(ins, 3, nw, on + 1, free);
        sqlite3_step(ins);
        sqlite3_reset(ins);
    }
    sqlite3_finalize(sel);
    sqlite3_finalize(ins);

    sqlite3_exec(db,
        "INSERT OR REPLACE INTO meta VALUES('ndocs',"
        "(SELECT COALESCE(MAX(id),0) FROM docs))", NULL, NULL, NULL);
    *rowid_out = did;
    t->data_version = -1;
    return SQLITE_OK;
}

static int sparse0_update(sqlite3_vtab *vt, int argc, sqlite3_value **argv,
                          sqlite3_int64 *rowid_out) {
    Sparse0Tab *t = (Sparse0Tab *)vt;
    if (argc == 1) {   /* DELETE: soft-delete, Python semantics */
        sqlite3_stmt *st;
        sqlite3_prepare_v2(t->db, "UPDATE docs SET deleted=1 WHERE id=?", -1, &st, NULL);
        sqlite3_bind_int64(st, 1, sqlite3_value_int64(argv[0]));
        sqlite3_step(st);
        sqlite3_finalize(st);
        t->data_version = -1;
        return SQLITE_OK;
    }
    if (sqlite3_value_type(argv[0]) != SQLITE_NULL) {
        t->base.zErrMsg = sqlite3_mprintf("sparse0: UPDATE not supported; DELETE then INSERT");
        return SQLITE_ERROR;
    }
    const char *text = (const char *)sqlite3_value_text(argv[2]);
    if (!text) {
        t->base.zErrMsg = sqlite3_mprintf("sparse0: text is required");
        return SQLITE_ERROR;
    }
    sqlite3_int64 rowid_in = sqlite3_value_type(argv[1]) == SQLITE_NULL
                             ? 0 : sqlite3_value_int64(argv[1]);
    return sparse0_insert_doc(t, rowid_in, text, rowid_out);
}

static sqlite3_module sparse0_module = {
    .iVersion = 0,
    .xCreate = sparse0_create,
    .xConnect = sparse0_connect,
    .xBestIndex = sparse0_bestindex,
    .xDisconnect = sparse0_disconnect,
    .xDestroy = sparse0_disconnect,
    .xOpen = sparse0_open,
    .xClose = sparse0_close,
    .xFilter = sparse0_filter,
    .xNext = sparse0_next,
    .xEof = sparse0_eof,
    .xColumn = sparse0_column,
    .xRowid = sparse0_rowid,
    .xUpdate = sparse0_update,
};

/* SQL functions */

static void fn_sparse_register(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    const char *name = (const char *)sqlite3_value_text(argv[0]);
    const char *gguf = (const char *)sqlite3_value_text(argv[1]);
    const char *sprs = (const char *)sqlite3_value_text(argv[2]);
    int max_seq = argc > 3 ? sqlite3_value_int(argv[3]) : 512;
    if (!name || !gguf || !sprs) { sqlite3_result_error(ctx, "sparse_register(name, gguf, sprs[, max_seq])", -1); return; }
    if (max_seq < 16 || max_seq > 512) { sqlite3_result_error(ctx, "max_seq must be in [16, 512]", -1); return; }
    SparseModel *m = model_find(name);
    if (m && strcmp(m->gguf_path, gguf) == 0 && strcmp(m->sidecar_sha, "") != 0) {
        char sha[65];
        if (sha256_file_hex(sprs, sha) == 0 && strcmp(sha, m->sidecar_sha) == 0) {
            if (m->max_seq != max_seq && m->enc) { sparse_encoder_free(m->enc); m->enc = NULL; }
            m->max_seq = max_seq;
            sqlite3_result_text(ctx, "ok", -1, SQLITE_STATIC);
            return;
        }
    }
    if (!m && g_n_models == MAX_MODELS) { sqlite3_result_error(ctx, "model registry full", -1); return; }
    SparseHead *h = sparse_head_load(sprs);
    if (!h) { sqlite3_result_error(ctx, "cannot load .sprs sidecar", -1); return; }
    if (m) {
        if (m->enc) { sparse_encoder_free(m->enc); m->enc = NULL; }
        free(m->gguf_path);
        /* earlier tables may still reference the old head; keep it */
    } else {
        m = &g_models[g_n_models++];
        m->name = strdup(name);
    }
    m->gguf_path = strdup(gguf);
    m->head = h;
    m->enc = NULL;
    m->n_threads = 4;
    m->max_seq = max_seq;
    if (sha256_file_hex(gguf, m->encoder_sha) != 0) strcpy(m->encoder_sha, "unreadable");
    sha256_file_hex(sprs, m->sidecar_sha);
    sha256_bytes_hex(h->vocab_blob, strlen(h->vocab_blob), m->vocab_sha);
    sqlite3_result_text(ctx, "ok", -1, SQLITE_STATIC);
}

/* Drop deleted ids from every posting list and remove their docs rows.
 * Returns the number of entries removed. */
static void fn_sparse_compact(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc; (void)argv;
    sqlite3 *db = sqlite3_context_db_handle(ctx);
    sqlite3_stmt *st;
    int n_dead = 0, cap = 0;
    int64_t *dead = NULL;
    if (sqlite3_prepare_v2(db, "SELECT id FROM docs WHERE deleted=1 ORDER BY id", -1, &st, NULL) != SQLITE_OK) {
        sqlite3_result_error(ctx, "sparse_compact: not a sqlite-sparse database", -1); return;
    }
    while (sqlite3_step(st) == SQLITE_ROW) {
        if (n_dead == cap) { cap = cap ? cap * 2 : 256; dead = realloc(dead, cap * sizeof(int64_t)); }
        dead[n_dead++] = sqlite3_column_int64(st, 0);
    }
    sqlite3_finalize(st);
    if (!n_dead) { free(dead); sqlite3_result_int64(ctx, 0); return; }
    sqlite3_exec(db, "SAVEPOINT sparse_compact", NULL, NULL, NULL);
    sqlite3_stmt *sel, *upd, *del;
    sqlite3_prepare_v2(db, "SELECT t, docs, ws FROM postings", -1, &sel, NULL);
    sqlite3_prepare_v2(db, "UPDATE postings SET docs=?, ws=? WHERE t=?", -1, &upd, NULL);
    sqlite3_prepare_v2(db, "DELETE FROM postings WHERE t=?", -1, &del, NULL);
    int64_t removed = 0;
    while (sqlite3_step(sel) == SQLITE_ROW) {
        int term = sqlite3_column_int(sel, 0);
        int n = sqlite3_column_bytes(sel, 1) / 4;
        const int32_t *d = sqlite3_column_blob(sel, 1);
        const uint8_t *w = sqlite3_column_blob(sel, 2);
        int32_t *nd = malloc((size_t)n * 4 + 4);
        uint8_t *nw = malloc((size_t)n + 1);
        int m = 0;
        for (int j = 0; j < n; j++) {

            int lo = 0, hi = n_dead - 1, hit = 0;
            while (lo <= hi) { int mid = (lo + hi) / 2; if (dead[mid] == d[j]) { hit = 1; break; } if (dead[mid] < d[j]) lo = mid + 1; else hi = mid - 1; }
            if (hit) { removed++; continue; }
            nd[m] = d[j]; nw[m] = w[j]; m++;
        }
        if (m == n) { free(nd); free(nw); continue; }
        if (m == 0) { sqlite3_bind_int(del, 1, term); sqlite3_step(del); sqlite3_reset(del); free(nd); free(nw); continue; }
        sqlite3_bind_blob(upd, 1, nd, m * 4, free);
        sqlite3_bind_blob(upd, 2, nw, m, free);
        sqlite3_bind_int(upd, 3, term);
        sqlite3_step(upd); sqlite3_reset(upd);
    }
    sqlite3_finalize(sel); sqlite3_finalize(upd); sqlite3_finalize(del);
    sqlite3_exec(db, "DELETE FROM docs WHERE deleted=1", NULL, NULL, NULL);
    sqlite3_exec(db, "RELEASE sparse_compact", NULL, NULL, NULL);
    free(dead);
    sqlite3_result_int64(ctx, removed);
}

static void json_append_string(sqlite3_str *s, const char *t) {
    sqlite3_str_appendchar(s, 1, '"');
    for (const unsigned char *p = (const unsigned char *)t; *p; p++) {
        if (*p == '"' || *p == '\\') { sqlite3_str_appendchar(s, 1, '\\'); sqlite3_str_appendchar(s, 1, (char)*p); }
        else if (*p < 0x20) sqlite3_str_appendf(s, "\\u%04x", *p);
        else sqlite3_str_appendchar(s, 1, (char)*p);
    }
    sqlite3_str_appendchar(s, 1, '"');
}

/* The query tokenizer's output for a string, as a JSON array of token strings,
 * using the vocabulary stored in this database. */
static void fn_sparse_tokens(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    const char *text = (const char *)sqlite3_value_text(argv[0]);
    if (!text) { sqlite3_result_null(ctx); return; }
    sqlite3 *db = sqlite3_context_db_handle(ctx);
    char *vjson = NULL;
    meta_get_text(db, "vocab", &vjson);
    if (!vjson) { sqlite3_result_error(ctx, "sparse_tokens: not a sqlite-sparse database", -1); return; }
    int n = 0;
    char *blob = json_vocab_to_blob(vjson, &n);
    sqlite3_free(vjson);
    if (!blob) { sqlite3_result_error(ctx, "sparse_tokens: vocabulary unreadable", -1); return; }
    Vocab *v = vocab_from_blob(blob);
    int32_t ids[MAX_QUERY_TERMS];
    int k = wp_tokenize(v, text, ids, MAX_QUERY_TERMS);
    sqlite3_str *s = sqlite3_str_new(db);
    sqlite3_str_appendchar(s, 1, '[');
    for (int i = 0; i < k; i++) {
        if (i) sqlite3_str_appendchar(s, 1, ',');
        json_append_string(s, v->entries[ids[i]].tok);
    }
    sqlite3_str_appendchar(s, 1, ']');
    vocab_free(v);
    sqlite3_result_text(ctx, sqlite3_str_finish(s), -1, sqlite3_free);
}

static void fn_sparse_version(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc; (void)argv;
    sqlite3_result_text(ctx, "sqlite-sparse/1 sparse0 0.1.0", -1, SQLITE_STATIC);
}

#ifdef _WIN32
__declspec(dllexport)
#endif
int sqlite3_sparse_init(sqlite3 *db, char **err, const sqlite3_api_routines *api) {
    (void)err;
    SQLITE_EXTENSION_INIT2(api);
    int rc = sqlite3_create_module(db, "sparse0", &sparse0_module, NULL);
    if (rc == SQLITE_OK)
        rc = sqlite3_create_function(db, "sparse_register", 3,
                                     SQLITE_UTF8 | SQLITE_DIRECTONLY, NULL,
                                     fn_sparse_register, NULL, NULL);
    if (rc == SQLITE_OK)
        rc = sqlite3_create_function(db, "sparse_register", 4,
                                     SQLITE_UTF8 | SQLITE_DIRECTONLY, NULL,
                                     fn_sparse_register, NULL, NULL);
    if (rc == SQLITE_OK)
        rc = sqlite3_create_function(db, "sparse_compact", 0,
                                     SQLITE_UTF8 | SQLITE_DIRECTONLY, NULL,
                                     fn_sparse_compact, NULL, NULL);
    if (rc == SQLITE_OK)
        rc = sqlite3_create_function(db, "sparse_tokens", 1,
                                     SQLITE_UTF8 | SQLITE_DIRECTONLY, NULL,
                                     fn_sparse_tokens, NULL, NULL);
    if (rc == SQLITE_OK)
        rc = sqlite3_create_function(db, "sparse_version", 0,
                                     SQLITE_UTF8 | SQLITE_DETERMINISTIC, NULL,
                                     fn_sparse_version, NULL, NULL);
    return rc;
}
