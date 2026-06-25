# Runtime Data

This directory is for local MVP runtime data:

- SQLite feedback database;
- API logs;
- temporary exports.

The generated `*.sqlite` files are intentionally ignored by git. Recreate the database with:

```bash
python3 tools/init_db.py
```
