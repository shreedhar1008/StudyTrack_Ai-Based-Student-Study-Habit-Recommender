import sqlite3
import json

conn = sqlite3.connect('studytrack.db')
cursor = conn.cursor()

# Get detailed breakdown
cursor.execute('''
    SELECT 
        risk_level,
        cluster_number,
        COUNT(*) as student_count,
        AVG(priority_score) as avg_priority
    FROM predictions
    WHERE id IN (SELECT MAX(id) FROM predictions GROUP BY student_id)
    GROUP BY risk_level, cluster_number
    ORDER BY 
        CASE risk_level 
            WHEN 'High' THEN 1 
            WHEN 'Moderate' THEN 2 
            WHEN 'Low' THEN 3 
        END,
        cluster_number
''')

results = []
for row in cursor.fetchall():
    results.append({
        'risk': row[0],
        'cluster': row[1],
        'count': row[2],
        'avg_priority': round(row[3], 2)
    })

# Print results
print(json.dumps(results, indent=2))

conn.close()
