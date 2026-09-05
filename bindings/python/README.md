# sqlite-sparse (Python)

Python binding for [sqlite-sparse](https://github.com/arbazsiddiqui/sqlite-sparse),
semantic search in one SQLite file with no model at query time.

```python
import sqlite3, sqlite_sparse
db = sqlite3.connect("notes.db")
sqlite_sparse.load(db)                 # loads the sparse0 extension
sqlite_sparse.register(db, "mini")     # downloads the model on first use
db.execute("CREATE VIRTUAL TABLE notes USING sparse0(model='mini')")
```

Also contains the pure-Python reference implementation of the file format
(`SparseIndex`) and the sidecar converter. Full documentation in the repository README.
