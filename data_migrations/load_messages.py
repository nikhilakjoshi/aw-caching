#!/usr/bin/env python3
"""
Data loader utility for copying askwealth.message table from server to local database.

This script connects to a remote Postgres database and copies all or filtered
rows from the askwealth.message table to a local Postgres database.
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any
from contextlib import contextmanager

from sqlalchemy import create_engine, text, Engine, Connection
from sqlalchemy.exc import SQLAlchemyError
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("data_migrations/load_messages.log"),
    ],
)
logger = logging.getLogger(__name__)

# Configuration
BATCH_SIZE = 1000
SERVER_DB_URL = os.getenv("SERVER_DB_URL")
LOCAL_DB_URL = os.getenv("LOCAL_DB_URL")


class MessageLoader:
    """Handles loading messages from server database to local database."""

    def __init__(self, server_db_url: str, local_db_url: str):
        """Initialize with database connection URLs."""
        if not server_db_url or not local_db_url:
            raise ValueError("Both SERVER_DB_URL and LOCAL_DB_URL must be provided")

        # Create engines with autocommit isolation level to avoid transaction issues
        self.server_engine = create_engine(server_db_url, isolation_level="AUTOCOMMIT")
        self.local_engine = create_engine(local_db_url, isolation_level="AUTOCOMMIT")
        self.stats = {"fetched": 0, "inserted": 0, "skipped": 0, "failed": 0}

    @contextmanager
    def get_connections(self):
        """Context manager for database connections."""
        server_conn = None
        local_conn = None
        try:
            server_conn = self.server_engine.connect()
            local_conn = self.local_engine.connect()
            yield server_conn, local_conn
        except SQLAlchemyError as e:
            logger.error(f"Database connection error: {e}")
            raise
        finally:
            if server_conn:
                server_conn.close()
            if local_conn:
                local_conn.close()

    def build_query(
        self,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        thread_id: Optional[str] = None,
    ) -> str:
        """Build SQL query with optional filters."""
        base_query = "SELECT * FROM askwealth.message"
        filters = []

        if from_date:
            filters.append(f"created_at >= '{from_date}'")
        if to_date:
            filters.append(f"created_at <= '{to_date}'")
        if thread_id:
            filters.append(f"thread_id = '{thread_id}'")

        if filters:
            base_query += " WHERE " + " AND ".join(filters)

        base_query += " ORDER BY created_at"
        return base_query

    def chunk_data(self, data: List[Any], size: int):
        """Yield successive chunks of data."""
        for i in range(0, len(data), size):
            yield data[i : i + size]

    def validate_row(self, row: Dict[str, Any]) -> bool:
        """Validate a row has required fields."""
        required_fields = ["message_id", "role", "created_at", "thread_id"]
        for field in required_fields:
            if field not in row or row[field] is None:
                logger.warning(f"Row missing required field '{field}': {row}")
                return False
        return True

    def insert_batch(
        self, conn: Connection, batch: List[Dict[str, Any]]
    ) -> tuple[int, int]:
        """Insert a batch of rows with upsert behavior."""
        inserted = 0
        failed = 0

        try:
            # Prepare batch data, filtering out invalid rows
            valid_rows = []
            for row_data in batch:
                if self.validate_row(row_data):
                    valid_rows.append(row_data)
                else:
                    failed += 1

            if not valid_rows:
                return inserted, failed

            # Use ON CONFLICT to handle duplicates
            insert_query = text(
                """
                INSERT INTO askwealth.message 
                (message_id, role, parts, created_at, thread_id, annotations)
                VALUES (:message_id, :role, :parts, :created_at, :thread_id, :annotations)
                ON CONFLICT (message_id) DO NOTHING
            """
            )

            # Execute with autocommit - no explicit transaction needed
            result = conn.execute(insert_query, valid_rows)
            inserted = result.rowcount

            logger.info(
                f"Batch processed: {inserted} inserted, {len(batch) - len(valid_rows)} invalid, "
                f"{len(valid_rows) - inserted} duplicates skipped"
            )

        except SQLAlchemyError as e:
            logger.error(f"Failed to insert batch: {e}")
            failed = len(batch)

        return inserted, failed

    def load_messages(
        self,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        thread_id: Optional[str] = None,
    ) -> Dict[str, int]:
        """Load messages from server to local database."""
        logger.info("Starting message loading process...")

        if from_date:
            logger.info(f"Filter: from_date >= {from_date}")
        if to_date:
            logger.info(f"Filter: to_date <= {to_date}")
        if thread_id:
            logger.info(f"Filter: thread_id = {thread_id}")

        query = self.build_query(from_date, to_date, thread_id)
        logger.info(f"Executing query: {query}")

        try:
            with self.get_connections() as (server_conn, local_conn):
                # Test connections
                server_conn.execute(text("SELECT 1"))
                local_conn.execute(text("SELECT 1"))
                logger.info("Database connections established successfully")

                # Fetch data from server
                result = server_conn.execute(text(query))
                rows = result.fetchall()
                self.stats["fetched"] = len(rows)

                logger.info(
                    f"Fetched {self.stats['fetched']} rows from server database"
                )

                if not rows:
                    logger.info("No rows to process")
                    return self.stats

                # Process in batches
                for batch_num, batch in enumerate(self.chunk_data(rows, BATCH_SIZE), 1):
                    logger.info(f"Processing batch {batch_num} ({len(batch)} rows)...")

                    # Convert rows to dictionaries
                    batch_data = [dict(row._mapping) for row in batch]

                    inserted, failed = self.insert_batch(local_conn, batch_data)

                    self.stats["inserted"] += inserted
                    self.stats["failed"] += failed
                    self.stats["skipped"] += len(batch) - inserted - failed

                logger.info("Loading completed successfully")

        except SQLAlchemyError as e:
            logger.error(f"Database error during loading: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during loading: {e}")
            raise

        return self.stats

    def print_summary(self):
        """Print summary statistics."""
        logger.info("=== LOADING SUMMARY ===")
        logger.info(f"Rows fetched from server: {self.stats['fetched']}")
        logger.info(f"Rows inserted to local:   {self.stats['inserted']}")
        logger.info(f"Rows skipped (duplicates): {self.stats['skipped']}")
        logger.info(f"Rows failed:              {self.stats['failed']}")
        logger.info("======================")


def validate_date_format(date_string: str) -> bool:
    """Validate date string is in YYYY-MM-DD format."""
    try:
        datetime.strptime(date_string, "%Y-%m-%d")
        return True
    except ValueError:
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Load messages from server database to local database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python data_migrations/load_messages.py
  python data_migrations/load_messages.py --from-date 2025-01-01
  python data_migrations/load_messages.py --from-date 2025-01-01 --to-date 2025-09-15
  python data_migrations/load_messages.py --thread-id "some-uuid-here"
        """,
    )

    parser.add_argument(
        "--from-date",
        help="Only load messages created after this date (YYYY-MM-DD format)",
    )
    parser.add_argument(
        "--to-date",
        help="Only load messages created before this date (YYYY-MM-DD format)",
    )
    parser.add_argument(
        "--thread-id", help="Only load messages for a specific thread ID"
    )

    args = parser.parse_args()

    # Validate date formats
    if args.from_date and not validate_date_format(args.from_date):
        logger.error("Invalid from-date format. Use YYYY-MM-DD")
        sys.exit(1)

    if args.to_date and not validate_date_format(args.to_date):
        logger.error("Invalid to-date format. Use YYYY-MM-DD")
        sys.exit(1)

    # Check environment variables
    if not SERVER_DB_URL:
        logger.error("SERVER_DB_URL environment variable is not set")
        sys.exit(1)

    if not LOCAL_DB_URL:
        logger.error("LOCAL_DB_URL environment variable is not set")
        sys.exit(1)

    try:
        # Create loader and run
        loader = MessageLoader(SERVER_DB_URL, LOCAL_DB_URL)
        stats = loader.load_messages(
            from_date=args.from_date, to_date=args.to_date, thread_id=args.thread_id
        )

        loader.print_summary()

        # Exit with appropriate code
        if stats["failed"] > 0:
            logger.warning(f"Completed with {stats['failed']} failures")
            sys.exit(1)
        else:
            logger.info("All operations completed successfully")
            sys.exit(0)

    except Exception as e:
        logger.error(f"Script failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
