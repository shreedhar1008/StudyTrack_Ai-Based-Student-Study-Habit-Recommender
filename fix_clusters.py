import sqlite3

# Connect to database
conn = sqlite3.connect('studytrack.db')
cursor = conn.cursor()

# Get all predictions with invalid clusters
cursor.execute('SELECT id, cluster_number FROM predictions WHERE cluster_number > 2')
invalid_clusters = cursor.fetchall()

print(f"Found {len(invalid_clusters)} predictions with invalid cluster numbers")

# Update invalid clusters to valid range (0-2)
for pred_id, cluster_num in invalid_clusters:
    # Map clusters 3 and 4 to valid range
    new_cluster = cluster_num % 3
    cursor.execute('UPDATE predictions SET cluster_number = ? WHERE id = ?', (new_cluster, pred_id))

conn.commit()

# Verify the fix
cursor.execute('SELECT DISTINCT cluster_number FROM predictions ORDER BY cluster_number')
distinct_clusters = [row[0] for row in cursor.fetchall()]
print(f"Distinct clusters after fix: {distinct_clusters}")

# Get cluster distribution
cursor.execute('''
    SELECT cluster_number, COUNT(*) 
    FROM predictions
    WHERE id IN (SELECT MAX(id) FROM predictions GROUP BY student_id)
    GROUP BY cluster_number
    ORDER BY cluster_number
''')
cluster_dist = cursor.fetchall()
print("\nCluster distribution (latest predictions):")
for cluster, count in cluster_dist:
    print(f"  Cluster {cluster}: {count} students")

conn.close()
print("\nDatabase cleaned successfully!")
