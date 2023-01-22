import json
from itertools import product
import math
import ebisu
import sqlite3
import os.path

file_exists = os.path.exists('mydatabase.db')
# Connect to or create a SQLite database
conn = sqlite3.connect('mydatabase.db')
cursor = conn.cursor()

learnTime = ebisu.timeMs() - 3600e3 * 100

if not file_exists:
  # Create a table with a JSON column
  cursor.execute('''
      CREATE TABLE mytable (
          id INTEGER PRIMARY KEY,
          json_column JSON
      )
  ''')

  # Insert some data into the table
  models = []
  for wmax in [.01, .1, .2, .3, .4, .5]:
    model = ebisu.initModel(wmax, now=learnTime)
    models.append(model)
    orig = model.to_json()
    cursor.execute("INSERT INTO mytable (json_column) VALUES (?)", (orig,))
  conn.commit()

query = """
SELECT f.id, (json_extract(value, '$[0]') - ( (?) - JSON_EXTRACT(json_column, '$.pred.lastEncounterMs'))  / (json_extract(value, '$[1]')))
FROM mytable f, json_each(json_extract(f.json_column, '$.pred.forSql'))
    """
queryGOOD = """
SELECT
  f.id,
  f.json_column,
  MAX(
    (
      json_extract(value, '$[0]') - (
        (?) - JSON_EXTRACT(json_column, '$.pred.lastEncounterMs')
      ) / json_extract(value, '$[1]')
    )
  )
FROM
  mytable f,
  json_each(json_extract(f.json_column, '$.pred.forSql'))
group by f.id
    """
query = queryGOOD

quizTime = ebisu.timeMs()
cursor.execute(query, (quizTime,))
results = cursor.fetchall()

for r in results:
  m = ebisu.Model.from_json(r[1])
  sql, py = r[2], ebisu.predictRecall(m, now=quizTime)
  print(sql, py, (sql - py) / py)

# Close the connection
conn.close()