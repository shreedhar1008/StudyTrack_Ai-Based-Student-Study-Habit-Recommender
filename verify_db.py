
import sqlite3

conn = sqlite3.connect('studytrack.db')
cursor = conn.cursor()

print("Verifying Test Users:")
try:
    cursor.execute("""
        SELECT s.full_name, p.risk_level, p.priority_score, p.cluster_number 
        FROM students s 
        JOIN predictions p ON s.student_id = p.student_id 
        WHERE s.full_name LIKE 'TestUser%'
        ORDER BY s.full_name
    """)
    
    rows = cursor.fetchall()
    if not rows:
        print("No TestUser records found!")
    else:
        for row in rows:
            print(row)
            
except Exception as e:
    print(e)
finally:
    conn.close()
