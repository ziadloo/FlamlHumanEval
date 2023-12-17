import json
import sqlite3
import os
from typing import Union


class ResultCache:
    def __init__(self, seed, cache_folder: str = "./.cache"):
        self.seed = seed
        self.cache_folder = cache_folder
        self.db_file = f"{self.cache_folder}/{seed}/cache.db"
        self.connection = None
        self.cursor = None
        self._initialize_database()

    def _initialize_database(self):
        # Check if the folder exists, create it if not
        if not os.path.exists(f"{self.cache_folder}/{self.seed}"):
            os.makedirs(f"{self.cache_folder}/{self.seed}")

        # Connect to the SQLite database
        self.connection = sqlite3.connect(self.db_file)
        self.cursor = self.connection.cursor()

        # Create the 'experiment' table if it doesn't exist
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS experiment (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key TEXT UNIQUE,
                result TEXT,
                metric_name TEXT,
                metric_value DECIMAL
            )
        ''')
        self.connection.commit()

    def close_connection(self):
        # Close the database connection
        if self.connection:
            self.connection.close()

    def check(self, config: dict) -> Union[float, None]:
        key = json.dumps(config, sort_keys=True)
        # Check if a record with the given key exists
        self.cursor.execute('SELECT metric_value FROM experiment WHERE key = ?', (key,))
        record = self.cursor.fetchone()
        return record[0] if record else None

    def store(self, config: dict, result: str, metric_name: str, metric_value: float) -> int:
        key = json.dumps(config, sort_keys=True)
        # Insert a new record into the 'experiment' table
        self.cursor.execute('''
            INSERT INTO experiment (key, result, metric_name, metric_value)
            VALUES (?, ?, ?, ?)
        ''', (key, result, metric_name, metric_value))
        self.connection.commit()

        # Return the id of the newly inserted record
        return self.cursor.lastrowid

    def update(self, id: int, result: str, metric_value: float) -> int:
        self.cursor.execute('''
            UPDATE experiment SET result=?, metric_value=? WHERE id=?
        ''', (result, metric_value, id))
        self.connection.commit()

    def restore(self):
        self.cursor.execute('''
            SELECT key, metric_value FROM experiment
        ''')
        keys = []
        values = []
        for record in self.cursor.fetchall():
            keys.append(json.loads(record[0]))
            values.append(record[1])

        return keys, values
