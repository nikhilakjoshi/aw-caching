# Data Migrations

This directory contains utilities for migrating data between databases, primarily for development and testing purposes.

## Scripts

### `load_messages.py`

A Python utility to copy the `askwealth.message` table from a remote server database to a local development database.

## Installation

### 1. Install Dependencies

All required dependencies are already included in the main project's `requirements.txt`. If you need to install them separately:

```bash
pip install sqlalchemy psycopg2-binary python-dotenv
```

### 2. Configure Environment Variables

Copy the `.env.example` file to `.env` and configure the database URLs:

```bash
cp .env.example .env
```

Edit the `.env` file and set the following variables:

```dotenv
# Server database (source) - the production/remote database
SERVER_DB_URL=postgresql://username:password@server_host:5432/server_database

# Local database (target) - your local development database
LOCAL_DB_URL=postgresql://username:password@localhost:5432/local_database
```

**Database URL Format:**

- `postgresql://[username[:password]]@[host[:port]]/[database_name]`
- For local databases, host is typically `localhost`
- Default PostgreSQL port is `5432`

### 3. Ensure Local Database Schema Exists

Before running the migration, make sure your local database has the `askwealth` schema and `message` table with the correct structure. You can use the existing Alembic migrations or create the schema manually.

## Usage

### Basic Usage

Load all messages from the server database:

```bash
python data_migrations/load_messages.py
```

### Filter by Date Range

Load messages created after a specific date:

```bash
python data_migrations/load_messages.py --from-date 2025-01-01
```

Load messages created within a date range:

```bash
python data_migrations/load_messages.py --from-date 2025-01-01 --to-date 2025-09-15
```

### Filter by Thread ID

Load messages for a specific thread:

```bash
python data_migrations/load_messages.py --thread-id "550e8400-e29b-41d4-a716-446655440000"
```

### Combined Filters

You can combine multiple filters:

```bash
python data_migrations/load_messages.py --from-date 2025-01-01 --thread-id "550e8400-e29b-41d4-a716-446655440000"
```

## Command Line Options

- `--from-date YYYY-MM-DD`: Only load messages created after this date
- `--to-date YYYY-MM-DD`: Only load messages created before this date
- `--thread-id UUID`: Only load messages for a specific thread ID

## Features

### Upsert Behavior

The script uses `ON CONFLICT (message_id) DO NOTHING` to handle duplicate records:

- If a message with the same `message_id` already exists locally → it will be skipped
- Otherwise → the new message will be inserted

### Batch Processing

- Processes data in batches of 1000 rows for optimal performance
- Reduces memory usage for large datasets
- Provides progress logging for long-running operations

### Error Handling

- Validates database connections before starting
- Validates required fields in each row
- Logs detailed error information for troubleshooting
- Continues processing even if individual rows fail
- Provides comprehensive statistics at completion

### Logging

The script logs to both:

- Standard output (console)
- Log file: `data_migrations/load_messages.log`

Log levels include:

- Progress updates during processing
- Row counts and batch information
- Error details for failed operations
- Final summary statistics

## Database Schema Requirements

The script expects the following table structure in both source and target databases:

```sql
CREATE TABLE askwealth.message (
    message_id UUID PRIMARY KEY,
    role VARCHAR NOT NULL,
    parts JSONB,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL,
    thread_id UUID NOT NULL,
    annotations JSONB
);
```

## Troubleshooting

### Connection Issues

1. **"SERVER_DB_URL environment variable is not set"**

   - Ensure your `.env` file exists and contains `SERVER_DB_URL`
   - Check that the `.env` file is in the project root directory

2. **"Database connection error"**
   - Verify database URLs are correct
   - Check that the databases are running and accessible
   - Ensure credentials have appropriate permissions

### Data Issues

1. **"Row missing required field"**

   - Source database may have incomplete data
   - Check the log file for specific row details
   - These rows will be skipped and counted as failures

2. **"No rows to process"**
   - Check your date filters - they might be too restrictive
   - Verify the source database contains data matching your criteria
   - Check that the `askwealth.message` table exists on the server

### Performance Issues

1. **Slow processing**

   - Large datasets may take time to process
   - Monitor the log output for progress updates
   - Consider using date filters to process data in smaller chunks

2. **Memory issues**
   - The script processes data in batches to limit memory usage
   - If you still encounter memory issues, consider processing smaller date ranges

## Example Output

```
2025-09-15 10:30:00,123 - INFO - Starting message loading process...
2025-09-15 10:30:00,125 - INFO - Filter: from_date >= 2025-01-01
2025-09-15 10:30:00,126 - INFO - Executing query: SELECT * FROM askwealth.message WHERE created_at >= '2025-01-01' ORDER BY created_at
2025-09-15 10:30:00,130 - INFO - Database connections established successfully
2025-09-15 10:30:02,456 - INFO - Fetched 2500 rows from server database
2025-09-15 10:30:02,458 - INFO - Processing batch 1 (1000 rows)...
2025-09-15 10:30:02,678 - INFO - Batch processed: 1000 inserted, 0 invalid, 0 duplicates skipped
2025-09-15 10:30:02,680 - INFO - Processing batch 2 (1000 rows)...
2025-09-15 10:30:02,895 - INFO - Batch processed: 995 inserted, 0 invalid, 5 duplicates skipped
2025-09-15 10:30:02,897 - INFO - Processing batch 3 (500 rows)...
2025-09-15 10:30:03,045 - INFO - Batch processed: 500 inserted, 0 invalid, 0 duplicates skipped
2025-09-15 10:30:03,046 - INFO - Loading completed successfully
2025-09-15 10:30:03,047 - INFO - === LOADING SUMMARY ===
2025-09-15 10:30:03,047 - INFO - Rows fetched from server: 2500
2025-09-15 10:30:03,047 - INFO - Rows inserted to local:   2495
2025-09-15 10:30:03,047 - INFO - Rows skipped (duplicates): 5
2025-09-15 10:30:03,047 - INFO - Rows failed:              0
2025-09-15 10:30:03,047 - INFO - ======================
```

## Security Considerations

- **Never commit your `.env` file** - it contains sensitive database credentials
- Use read-only database credentials for the server connection when possible
- Consider using connection pooling for production environments
- Regularly rotate database credentials

## Future Enhancements

Planned improvements include:

- **Dry-run mode**: Preview what would be copied without actually inserting data
- **Additional tables**: Support for copying other tables like `document`, `chunk`, etc.
- **Parallel processing**: Speed up large dataset migrations
- **CSV export/import**: Alternative workflow using CSV as intermediate format
- **Incremental sync**: Only copy new/modified records since last sync
- **Backup creation**: Automatically backup local data before overwriting

## Contributing

When adding new migration scripts:

1. Follow the same error handling and logging patterns
2. Use environment variables for configuration
3. Include comprehensive CLI help and documentation
4. Add appropriate tests
5. Update this README with new script documentation
