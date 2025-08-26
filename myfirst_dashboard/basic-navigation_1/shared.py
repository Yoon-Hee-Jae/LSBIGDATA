from pathlib import Path
import sqlite3
import pandas as pd

app_dir = Path(__file__).parent
conn = sqlite3.connect("./data/penguins.db")

df = pd.read_sql_query("SELECT * FROM penguins;", conn)
