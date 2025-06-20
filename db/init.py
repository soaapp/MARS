import os
import sqlite3
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('MARS')

# Get the absolute path to the database file
db_dir = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(db_dir, "mock_data.db")

logger.info(f"Creating database at: {db_path}")

# Create the database and tables
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Drop the table if it exists to start fresh
cursor.execute("DROP TABLE IF EXISTS people")

# Create the people table
cursor.execute("CREATE TABLE IF NOT EXISTS people (id INTEGER, name TEXT, role TEXT)")

# Insert sample data
cursor.executemany("INSERT INTO people VALUES (?, ?, ?)", [
    (1, "Alice", "Engineer"),
    (2, "Bob", "Manager"),
    (3, "Charlie", "Developer"),
    (4, "Diana", "Designer")
])

# Commit changes and close connection
conn.commit()
conn.close()

logger.info(f"Database initialized successfully with employee records")

# Verify the database was created
if os.path.exists(db_path):
    logger.info(f"Database file confirmed at: {db_path}")
    # Get file size
    size = os.path.getsize(db_path)
    logger.info(f"Database size: {size} bytes")
else:
    logger.error(f"Database file not found at: {db_path}")

