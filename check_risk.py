import sqlite3

conn = sqlite3.connect('studytrack.db')
cursor = conn.cursor()

# Get risk distribution
cursor.execute('''
    SELECT risk_level, COUNT(*) 
    FROM predictions
    WHERE id IN (SELECT MAX(id) FROM predictions GROUP BY student_id)
    GROUP BY risk_level
''')

print("Risk Distribution (latest predictions):")
for row in cursor.fetchall():
    print(f"  {row[0]}: {row[1]} students")

conn.close()
