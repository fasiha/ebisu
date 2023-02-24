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
  for initHl in [1, 5, 25, 125]:
    model = ebisu.initModel(initHl, now=learnTime)
    models.append(model)
    orig = model.to_json()
    cursor.execute("INSERT INTO mytable (json_column) VALUES (?)", (orig,))
  conn.commit()

query = """
SELECT
  t.id,
  t.json_column,
  SUM(
    pow(2,
      json_extract(value, '$[0]') - (
        (?) - json_extract(json_column, '$.pred.lastEncounterMs')
      ) * json_extract(value, '$[1]')
    )
  ) AS logPredictRecall
FROM
  mytable t,
  json_each(json_extract(t.json_column, '$.pred.forSql'))
GROUP BY t.id
    """

quizTime = ebisu.timeMs()
cursor.execute(query, (quizTime,))
results = cursor.fetchall()

for r in results:
  m = ebisu.Model.from_json(r[1])
  sql = r[2]**(1 / m.pred.power)
  py = ebisu.predictRecall(m, now=quizTime, logDomain=False)
  print(sql, py, (sql - py) / py)

# Close the connection
conn.close()