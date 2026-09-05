/* WordPiece tokenizer with BERT's basic tokenization in front of it: control
 * characters dropped, CJK ideographs isolated, split on whitespace, lowercase,
 * combining marks removed (NFD, category Mn), punctuation split off, then greedy
 * longest match with ## continuations. A word that fails to tokenize, or is
 * longer than 100 characters, becomes [UNK], as in the reference tokenizer. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "utf8proc.h"

#define MAX_TOKEN 512
#define MAX_WORD_CHARS 100

typedef struct { char *tok; int32_t id; } Entry;

typedef struct {
    Entry **table;    /* open addressing, power-of-two slots */
    uint32_t mask;
    char *blob;
    Entry *entries;
    int n;
    int32_t unk;      /* id of [UNK], or -1 */
} Vocab;

static uint32_t fnv1a(const char *s, size_t len) {
    uint32_t h = 2166136261u;
    for (size_t i = 0; i < len; i++) { h ^= (uint8_t)s[i]; h *= 16777619u; }
    return h;
}

static int32_t vocab_lookup(const Vocab *v, const char *s, size_t len) {
    if (len == 0 || len >= MAX_TOKEN) return -1;
    char buf[MAX_TOKEN];
    memcpy(buf, s, len); buf[len] = 0;
    uint32_t h = fnv1a(buf, len) & v->mask;
    while (v->table[h]) {
        if (strcmp(v->table[h]->tok, buf) == 0) return v->table[h]->id;
        h = (h + 1) & v->mask;
    }
    return -1;
}

static void vocab_index(Vocab *v) {
    uint32_t slots = 1024;
    while (slots < (uint32_t)v->n * 4) slots <<= 1;
    v->mask = slots - 1;
    v->table = calloc(slots, sizeof(Entry *));
    for (int i = 0; i < v->n; i++) {
        uint32_t h = fnv1a(v->entries[i].tok, strlen(v->entries[i].tok)) & v->mask;
        while (v->table[h]) h = (h + 1) & v->mask;
        v->table[h] = &v->entries[i];
    }
    v->unk = vocab_lookup(v, "[UNK]", 5);
}

/* takes ownership of blob, newline-separated tokens */
static Vocab *vocab_from_blob(char *blob) {
    Vocab *v = calloc(1, sizeof(Vocab));
    v->blob = blob;
    int cap = 40000;
    v->entries = malloc(cap * sizeof(Entry));
    char *line = v->blob;
    for (char *p = v->blob; ; p++) {
        if (*p == '\n' || *p == 0) {
            int end = (*p == 0);
            *p = 0;
            if (v->n == cap) { cap *= 2; v->entries = realloc(v->entries, cap * sizeof(Entry)); }
            v->entries[v->n].tok = line;
            v->entries[v->n].id = v->n;
            v->n++;
            if (end) break;
            line = p + 1;
        }
    }
    if (v->n && v->entries[v->n-1].tok[0] == 0) v->n--;   /* trailing newline */
    vocab_index(v);
    return v;
}

static Vocab *vocab_load(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    char *blob = malloc(sz + 1);
    fread(blob, 1, sz, f); blob[sz] = 0; fclose(f);
    return vocab_from_blob(blob);
}

static void vocab_free(Vocab *v) {
    if (!v) return;
    free(v->table); free(v->entries); free(v->blob); free(v);
}

static int is_cjk(int32_t cp) {
    return (cp >= 0x4E00 && cp <= 0x9FFF) || (cp >= 0x3400 && cp <= 0x4DBF)
        || (cp >= 0x20000 && cp <= 0x2A6DF) || (cp >= 0x2A700 && cp <= 0x2B73F)
        || (cp >= 0x2B740 && cp <= 0x2B81F) || (cp >= 0x2B820 && cp <= 0x2CEAF)
        || (cp >= 0xF900 && cp <= 0xFAFF) || (cp >= 0x2F800 && cp <= 0x2FA1F);
}

static int is_punct(int32_t cp) {
    if ((cp >= 33 && cp <= 47) || (cp >= 58 && cp <= 64) || (cp >= 91 && cp <= 96) || (cp >= 123 && cp <= 126)) return 1;
    utf8proc_category_t c = utf8proc_category(cp);
    return c >= UTF8PROC_CATEGORY_PC && c <= UTF8PROC_CATEGORY_PO;
}

static int is_control(utf8proc_category_t c) {
    return c == UTF8PROC_CATEGORY_CC || c == UTF8PROC_CATEGORY_CF || c == UTF8PROC_CATEGORY_CN
        || c == UTF8PROC_CATEGORY_CS || c == UTF8PROC_CATEGORY_CO;
}

static int is_sep(int32_t cp) {
    if (cp == ' ') return 1;
    utf8proc_category_t c = utf8proc_category(cp);
    return c == UTF8PROC_CATEGORY_ZL || c == UTF8PROC_CATEGORY_ZP;
}

static int emit(const Vocab *v, int32_t id, int32_t *out, int nout, int max) {
    if (id >= 0 && nout < max) out[nout++] = id;
    return nout;
}

/* one punctuation-free word as codepoints -> word pieces (or [UNK]) */
static int wp_piece(const Vocab *v, const int32_t *cps, int n, int32_t *out, int nout, int max) {
    if (n > MAX_WORD_CHARS) return emit(v, v->unk, out, nout, max);
    uint8_t *bytes = malloc((size_t)n * 4 + 8);
    int *off = malloc((size_t)(n + 1) * sizeof(int));
    int32_t *ids = malloc((size_t)n * sizeof(int32_t));
    int total = 0;
    for (int i = 0; i < n; i++) {
        off[i] = total;
        total += (int)utf8proc_encode_char(cps[i], bytes + total);
    }
    off[n] = total;
    char cand[MAX_TOKEN];
    int s = 0, k = 0;
    while (s < n) {
        int e = n, hit = -1;
        for (; e > s; e--) {
            int len = off[e] - off[s];
            int clen = s ? len + 2 : len;
            if (clen >= MAX_TOKEN) continue;
            if (s) { cand[0] = '#'; cand[1] = '#'; memcpy(cand + 2, bytes + off[s], len); }
            else memcpy(cand, bytes + off[s], len);
            hit = vocab_lookup(v, cand, clen);
            if (hit >= 0) break;
        }
        if (hit < 0) { k = -1; break; }
        ids[k++] = hit;
        s = e;
    }
    if (k < 0) nout = emit(v, v->unk, out, nout, max);
    else for (int i = 0; i < k; i++) nout = emit(v, ids[i], out, nout, max);
    free(bytes); free(off); free(ids);
    return nout;
}

/* one whitespace-delimited word: lowercase, strip marks, split punctuation */
static int wp_word(const Vocab *v, const int32_t *cps, int n, int32_t *out, int nout, int max) {
    int32_t *buf = malloc(((size_t)n * 4 + 8) * sizeof(int32_t));
    int m = 0;
    for (int i = 0; i < n; i++) {
        int32_t dec[8];
        int bc = 0;
        utf8proc_ssize_t k = utf8proc_decompose_char(utf8proc_tolower(cps[i]), dec, 8, UTF8PROC_DECOMPOSE, &bc);
        if (k <= 0 || k > 8) { dec[0] = utf8proc_tolower(cps[i]); k = 1; }
        for (int j = 0; j < k; j++)
            if (utf8proc_category(dec[j]) != UTF8PROC_CATEGORY_MN) buf[m++] = dec[j];
    }
    int i = 0;
    while (i < m && nout < max) {
        if (is_punct(buf[i])) { nout = wp_piece(v, buf + i, 1, out, nout, max); i++; continue; }
        int j = i;
        while (j < m && !is_punct(buf[j])) j++;
        nout = wp_piece(v, buf + i, j - i, out, nout, max);
        i = j;
    }
    free(buf);
    return nout;
}

static int wp_tokenize(const Vocab *v, const char *text, int32_t *out, int max_ids) {
    size_t len = strlen(text);
    int32_t *cps = malloc((len * 3 + 8) * sizeof(int32_t));
    int n = 0;
    size_t i = 0;
    while (i < len) {
        int32_t cp;
        utf8proc_ssize_t adv = utf8proc_iterate((const utf8proc_uint8_t *)text + i, (utf8proc_ssize_t)(len - i), &cp);
        if (adv <= 0 || cp < 0) { i++; continue; }
        i += (size_t)adv;
        if (cp == 0 || cp == 0xFFFD) continue;
        if (cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r') { cps[n++] = ' '; continue; }
        utf8proc_category_t cat = utf8proc_category(cp);
        if (cat == UTF8PROC_CATEGORY_ZS) { cps[n++] = ' '; continue; }
        if (is_control(cat)) continue;
        if (is_cjk(cp)) { cps[n++] = ' '; cps[n++] = cp; cps[n++] = ' '; continue; }
        cps[n++] = cp;
    }
    int nout = 0, s = 0;
    while (s < n && nout < max_ids) {
        while (s < n && is_sep(cps[s])) s++;
        int e = s;
        while (e < n && !is_sep(cps[e])) e++;
        if (e > s) nout = wp_word(v, cps + s, e - s, out, nout, max_ids);
        s = e;
    }
    free(cps);
    return nout;
}

#ifdef WP_MAIN
int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s vocab.txt < lines\n", argv[0]); return 1; }
    Vocab *v = vocab_load(argv[1]);
    char line[8192];
    int32_t ids[512];
    while (fgets(line, sizeof(line), stdin)) {
        size_t l = strlen(line);
        if (l && line[l-1] == '\n') line[l-1] = 0;
        int n = wp_tokenize(v, line, ids, 512);
        for (int i = 0; i < n; i++) printf(i ? " %d" : "%d", ids[i]);
        printf("\n");
    }
    return 0;
}
#endif
