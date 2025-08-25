import sqlite3
import pandas as pd

# db 파일 연결
conn = sqlite3.connect("penguins.db")

df = pd.read_sql_query("SELECT * FROM penguins LIMIT 5;", conn)


