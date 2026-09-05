/* Scatter-add scoring over posting lists, then top-k over the documents
 * touched. The score buffer and prepared statement persist in a ScoreCtx. */
#include <sqlite3.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { int32_t doc; float score; } Hit;

typedef struct {
    float *score;        /* [cap+1], zero except for touched docs */
    int32_t *touched;    /* [cap+1] doc ids with a nonzero score */
    int cap;
    sqlite3_stmt *st;    /* cached "SELECT docs, ws FROM postings WHERE t=?" */
    sqlite3 *st_db;
} ScoreCtx;

static void score_ctx_free(ScoreCtx *c) {
    if (!c) return;
    if (c->st) sqlite3_finalize(c->st);
    free(c->score);
    free(c->touched);
    memset(c, 0, sizeof(*c));
}

static int score_ctx_ensure(ScoreCtx *c, sqlite3 *db, int ndocs) {
    if (ndocs > c->cap) {
        float *s = realloc(c->score, (size_t)(ndocs + 1) * sizeof(float));
        int32_t *t = realloc(c->touched, (size_t)(ndocs + 1) * sizeof(int32_t));
        if (!s || !t) return -1;

        memset(s + (c->cap ? c->cap + 1 : 0), 0,
               (size_t)(ndocs - (c->cap ? c->cap : -1)) * sizeof(float));
        c->score = s;
        c->touched = t;
        c->cap = ndocs;
    }
    if (c->st && c->st_db != db) { sqlite3_finalize(c->st); c->st = NULL; }
    if (!c->st) {
        if (sqlite3_prepare_v2(db, "SELECT docs, ws FROM postings WHERE t=?", -1,
                               &c->st, NULL) != SQLITE_OK) return -1;
        c->st_db = db;
    }
    return 0;
}

/* insert into a descending top-k buffer holding n entries */
static int topk_insert(Hit *out, int n, int k, int32_t doc, float s) {
    int i = n < k ? n : k - 1;
    if (n == k && s <= out[k - 1].score) return n;
    while (i > 0 && out[i - 1].score < s) { out[i] = out[i - 1]; i--; }
    out[i].doc = doc;
    out[i].score = s;
    return n < k ? n + 1 : k;
}

static int sparse_score_ctx(ScoreCtx *c, sqlite3 *db, const int32_t *qterms,
                            const float *qweights, int nq, int ndocs,
                            float weight_scale, int weight_mode_u8,
                            int k, Hit *out) {
    if (score_ctx_ensure(c, db, ndocs) != 0) return -1;
    float *score = c->score;
    int32_t *touched = c->touched;
    int n_touched = 0;
    const float inv_scale = 1.0f / weight_scale;

    for (int i = 0; i < nq; i++) {
        sqlite3_bind_int(c->st, 1, qterms[i]);
        if (sqlite3_step(c->st) == SQLITE_ROW) {
            int n = sqlite3_column_bytes(c->st, 0) / 4;
            const int32_t *docs = sqlite3_column_blob(c->st, 0);
            const void *ws = sqlite3_column_blob(c->st, 1);
            float qw = qweights[i];
            if (weight_mode_u8) {
                const uint8_t *w = ws;
                for (int j = 0; j < n; j++) {
                    int32_t d = docs[j];
                    if (d < 1 || d > ndocs) continue;
                    if (score[d] == 0.0f) touched[n_touched++] = d;
                    score[d] += qw * (w[j] * inv_scale);
                }
            } else {
                const float *w = ws;
                for (int j = 0; j < n; j++) {
                    int32_t d = docs[j];
                    if (d < 1 || d > ndocs) continue;
                    if (score[d] == 0.0f) touched[n_touched++] = d;
                    score[d] += qw * w[j];
                }
            }
        }
        sqlite3_reset(c->st);
    }

    int nk = 0;
    for (int i = 0; i < n_touched; i++) {
        int32_t d = touched[i];
        float s = score[d];
        if (s > 0.0f) nk = topk_insert(out, nk, k, d, s);
        score[d] = 0.0f;
    }
    return nk;
}

int sparse_score(sqlite3 *db, const int32_t *qterms, const float *qweights,
                 int nq, int ndocs, float weight_scale, int weight_mode_u8,
                 int k, Hit *out) {
    ScoreCtx c;
    memset(&c, 0, sizeof(c));
    int n = sparse_score_ctx(&c, db, qterms, qweights, nq, ndocs,
                             weight_scale, weight_mode_u8, k, out);
    score_ctx_free(&c);
    return n;
}

#ifdef SCORER_MAIN
#include "wordpiece.c"
int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s index.db 'query text'\n", argv[0]); return 1; }
    sqlite3 *db;
    if (sqlite3_open_v2(argv[1], &db, SQLITE_OPEN_READONLY, NULL)) { fprintf(stderr, "open failed\n"); return 1; }
    /* meta */
    sqlite3_stmt *st;
    int ndocs = 0, u8 = 1;
    double scale = 40.0;
    char *vocab_tmp = "/tmp/sparse_vocab.txt";
    sqlite3_prepare_v2(db, "SELECT k, v FROM meta", -1, &st, NULL);
    while (sqlite3_step(st) == SQLITE_ROW) {
        const char *k = (const char *)sqlite3_column_text(st, 0);
        const char *v = (const char *)sqlite3_column_text(st, 1);
        if (!strcmp(k, "ndocs")) ndocs = atoi(v);
        else if (!strcmp(k, "weight_scale")) scale = atof(v);
        else if (!strcmp(k, "weight_mode")) u8 = !strcmp(v, "u8");
        else if (!strcmp(k, "vocab")) {           /* JSON array -> plain lines */
            FILE *f = fopen(vocab_tmp, "w");
            const char *p = v;
            while (*p) {
                if (*p == '"') {
                    p++;
                    while (*p && *p != '"') { if (*p == '\\') p++; fputc(*p++, f); }
                    fputc('\n', f);
                }
                p++;
            }
            fclose(f);
        }
    }
    sqlite3_finalize(st);
    Vocab *vc = vocab_load(vocab_tmp);
    int32_t ids[256];
    int nids = wp_tokenize(vc, argv[2], ids, 256);

    int32_t qterms[256]; float qweights[256]; int nq = 0;
    sqlite3_prepare_v2(db, "SELECT w FROM qlut WHERE t=?", -1, &st, NULL);
    for (int i = 0; i < nids; i++) {
        sqlite3_bind_int(st, 1, ids[i]);
        if (sqlite3_step(st) == SQLITE_ROW) {
            double w = sqlite3_column_double(st, 0);
            int found = 0;
            for (int j = 0; j < nq; j++) if (qterms[j] == ids[i]) { qweights[j] += w; found = 1; }
            if (!found) { qterms[nq] = ids[i]; qweights[nq] = w; nq++; }
        }
        sqlite3_reset(st);
    }
    sqlite3_finalize(st);
    Hit hits[100];
    int n = sparse_score(db, qterms, qweights, nq, ndocs, scale, u8, 10, hits);
    for (int i = 0; i < n; i++) printf("%d %.6f\n", hits[i].doc, hits[i].score);
    return 0;
}
#endif
