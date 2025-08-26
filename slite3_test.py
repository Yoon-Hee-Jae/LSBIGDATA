import sqlite3
import pandas as pd

# db 파일 연결
conn = sqlite3.connect("./data/penguins.db")

df = pd.read_sql_query("SELECT * FROM penguins LIMIT 5;", conn)
df

df2 = pd.DataFrame({
    "name": ["Alice", "Bob"],
    "age": [25, 30]
})
# SQLite 테이블로 저장
df2.to_sql("people", conn, if_exists="replace", index=False)

conn.close()
