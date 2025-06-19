import sqlite3
conn = sqlite3.connect("mock_data.db")
cursor = conn.cursor()
cursor.execute("CREATE TABLE IF NOT EXISTS people (id INTEGER, name TEXT, role TEXT)")
cursor.executemany("INSERT INTO people VALUES (?, ?, ?)", [
    (1, "Alice", "Engineer"),
    (2, "Bob", "Manager")
])
conn.commit()
conn.close()
