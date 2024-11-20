"""
Simple script to interact and modify the cache settings
"""

import sqlite3
import pandas as pd
from diskcache import Cache

# Connnect and reset cache settings
CACHE_PATH = "embedding_cache"
cache = Cache(CACHE_PATH)
cache.reset('size_limit', (10 * 1024**3))  # 10 GiB

# Connect to the database and print settings table
conn = sqlite3.connect('embedding_cache/cache.db')
tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)
print("Tables:", tables)
TABLE_NAME = 'Settings'
data = pd.read_sql_query(f"SELECT * FROM {TABLE_NAME};", conn)
conn.close()

print(data)
