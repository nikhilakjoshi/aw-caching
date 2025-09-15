# Requirements Document: Data Loader Utility for `askwealth.message`

## Objective

We need a Python-based script/utility to **copy the `askwealth.message` table from the server database into a local database** (development environment).
This will allow local development and testing against real data without directly querying production.

---

## Source and Target

- **Source Database (Server)**

  - Postgres instance
  - Schema: `askwealth`
  - Table: `message`
  - Authentication: username, password, host, port, database (to be provided via config or environment variables)

- **Target Database (Local)**

  - Postgres instance running locally
  - Schema: `askwealth` (already created locally)
  - Table: `message` (structure matches server schema)

---

## Requirements

### 1. Python Utility

- Must be written in Python 3.
- Use **SQLAlchemy** (preferred) or `psycopg2` for DB connections.
- Must allow configuration via:

  - `.env` file (preferred)
  - or environment variables (e.g., `SERVER_DB_URL`, `LOCAL_DB_URL`).

- All scripts must live under a dedicated **`data_migrations/` directory** inside the project.

### 2. Functionality

1. **Connect to Source DB (server)**.
2. **Connect to Target DB (local)**.
3. **Fetch all rows** from `askwealth.message`.

   - Optionally support filtering by `created_at` (date range) or `thread_id`.

4. **Insert into local DB**:

   - Upsert behavior:

     - If a `message_id` already exists locally → skip or update.
     - Otherwise → insert new row.

   - Preserve UUIDs, JSONB, and timestamps exactly as on server.

5. **Logging**:

   - Log row counts fetched, inserted, skipped, and failed.
   - Write to stdout and optionally to a file.

6. **Error Handling**:

   - Catch and log connection issues.
   - Catch and log insertion failures (with row info).

7. **Performance**:

   - Use batch inserts (e.g., 1000 rows at a time).
   - Avoid row-by-row inserts to keep speed reasonable.

### 3. Script Interface

- Run as CLI tool:

  ```bash
  python data_migrations/load_messages.py --from-date 2025-01-01 --to-date 2025-09-15
  ```

- Arguments:

  - `--from-date` (optional): only load messages created after this date.
  - `--to-date` (optional): only load messages created before this date.
  - `--thread-id` (optional): only load messages for a specific thread.
  - If no args → load full table.

### 4. Example Python Code Skeleton

```python
import os
import argparse
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

def load_messages(from_date=None, to_date=None, thread_id=None):
    server_engine = create_engine(os.getenv("SERVER_DB_URL"))
    local_engine = create_engine(os.getenv("LOCAL_DB_URL"))

    filters = []
    if from_date:
        filters.append(f"created_at >= '{from_date}'")
    if to_date:
        filters.append(f"created_at <= '{to_date}'")
    if thread_id:
        filters.append(f"thread_id = '{thread_id}'")

    where_clause = "WHERE " + " AND ".join(filters) if filters else ""
    query = f"SELECT * FROM askwealth.message {where_clause}"

    with server_engine.connect() as src_conn, local_engine.begin() as dest_conn:
        result = src_conn.execute(text(query))
        rows = result.fetchall()

        for batch in chunk(rows, 1000):
            dest_conn.execute(
                text("""
                    INSERT INTO askwealth.message (message_id, role, parts, created_at, thread_id, annotations)
                    VALUES (:message_id, :role, :parts, :created_at, :thread_id, :annotations)
                    ON CONFLICT (message_id) DO NOTHING
                """),
                [dict(row._mapping) for row in batch]
            )

def chunk(data, size):
    for i in range(0, len(data), size):
        yield data[i:i+size]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-date")
    parser.add_argument("--to-date")
    parser.add_argument("--thread-id")
    args = parser.parse_args()

    load_messages(args.from_date, args.to_date, args.thread_id)
```

---

## Deliverables

1. `data_migrations/load_messages.py` script:

   - Implements functionality described above.
   - Configurable via `.env` or env vars.
   - Supports CLI filters.
   - Handles upserts correctly.

2. `README.md` inside `data_migrations/` with:

   - How to install dependencies.
   - How to configure `.env` with DB URLs.
   - Example CLI usage.

---

## Future Enhancements

- Add support for syncing other tables (`document`, `chunk`, etc.).
- Add `--dry-run` option to preview rows without inserting.
- Add support for parallel loading for very large datasets.
- Add option to dump to CSV as intermediate format before load.
