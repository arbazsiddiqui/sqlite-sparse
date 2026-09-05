"""CLI: sqlite-sparse build|search|sync|attach|info"""
import argparse
import json
import sys


def main():
    p = argparse.ArgumentParser(prog="sqlite-sparse")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="index a JSONL file of {id, text[, title]}")
    b.add_argument("db"); b.add_argument("jsonl"); b.add_argument("--model", default="mini")
    s = sub.add_parser("search"); s.add_argument("db"); s.add_argument("query"); s.add_argument("-k", type=int, default=10)
    y = sub.add_parser("sync"); y.add_argument("db")
    a = sub.add_parser("attach"); a.add_argument("db"); a.add_argument("table"); a.add_argument("columns", help="comma-separated"); a.add_argument("--id-col", default="rowid"); a.add_argument("--model", default="mini")
    i = sub.add_parser("info"); i.add_argument("db")
    c = sub.add_parser("convert", help="write the .sprs sidecar for an inference-free HF checkpoint")
    c.add_argument("model_id"); c.add_argument("out")
    c.add_argument("--double-log", action="store_true", help="v3 models (log1p applied twice)")
    args = p.parse_args()
    from .api import SparseIndex

    if args.cmd == "build":
        ix = SparseIndex(args.db, model=args.model)
        for line in open(args.jsonl):
            r = json.loads(line)
            ix.add(id=r["id"], text=r["text"], title=r.get("title", ""))
        n = ix.commit()
        print(f"indexed {n} docs -> {args.db}")
    elif args.cmd == "search":
        ix = SparseIndex(args.db)
        for r in ix.search(args.query, k=args.k):
            print(f"{r['score']:8.3f}  {r['ext_id']}  {r['title'][:70]}")
    elif args.cmd == "sync":
        print(f"synced {SparseIndex(args.db).sync()} docs")
    elif args.cmd == "attach":
        ix = SparseIndex(args.db, model=args.model)
        ix.attach(args.table, args.columns.split(","), id_col=args.id_col)
        print(f"attached to {args.table}({args.columns}); synced {ix.sync()} rows")
    elif args.cmd == "convert":
        from .convert import extract
        extract(args.model_id, args.out, act_flag=1 if args.double_log else 0)
    elif args.cmd == "info":
        ix = SparseIndex(args.db)
        st = ix.store
        print(json.dumps({"model": st.get_meta("model_id"), "ndocs": st.get_meta("ndocs"),
                          "pending": st.pending_count(), "weight_mode": st.get_meta("weight_mode")}, indent=2))


if __name__ == "__main__":
    sys.exit(main())
