import sqlite3

conn = sqlite3.connect('studytrack.db')
cursor = conn.cursor()

# Get detailed breakdown of risk, cluster, and priority
cursor.execute('''
    SELECT 
        risk_level,
        cluster_number,
        priority_score,
        COUNT(*) as student_count
    FROM predictions
    WHERE id IN (SELECT MAX(id) FROM predictions GROUP BY student_id)
    GROUP BY risk_level, cluster_number, priority_score
    ORDER BY risk_level DESC, cluster_number, priority_score DESC
''')

print("=" * 80)
print("DETAILED STUDENT DISTRIBUTION")
print("=" * 80)
print(f"{'Risk Level':<15} {'Cluster':<12} {'Priority':<12} {'Students':<12}")
print("-" * 80)

current_risk = None
for row in cursor.fetchall():
    risk, cluster, priority, count = row
    
    if risk != current_risk:
        if current_risk is not None:
            print("-" * 80)
        current_risk = risk
    
    print(f"{risk:<15} Cluster {cluster:<5} {priority}/15{'':<7} {count:<12}")

print("=" * 80)

# Summary by risk level
cursor.execute('''
    SELECT risk_level, COUNT(*) as count
    FROM predictions
    WHERE id IN (SELECT MAX(id) FROM predictions GROUP BY student_id)
    GROUP BY risk_level
    ORDER BY risk_level DESC
''')

print("\nRISK LEVEL SUMMARY:")
print("-" * 40)
for row in cursor.fetchall():
    print(f"  {row[0]}: {row[1]:,} students")

# Summary by cluster
cursor.execute('''
    SELECT cluster_number, COUNT(*) as count
    FROM predictions
    WHERE id IN (SELECT MAX(id) FROM predictions GROUP BY student_id)
    GROUP BY cluster_number
    ORDER BY cluster_number
''')

print("\nCLUSTER SUMMARY:")
print("-" * 40)
for row in cursor.fetchall():
    print(f"  Cluster {row[0]}: {row[1]:,} students")

# Average priority by risk and cluster
cursor.execute('''
    SELECT 
        risk_level,
        cluster_number,
        AVG(priority_score) as avg_priority,
        COUNT(*) as count
    FROM predictions
    WHERE id IN (SELECT MAX(id) FROM predictions GROUP BY student_id)
    GROUP BY risk_level, cluster_number
    ORDER BY risk_level DESC, cluster_number
''')

print("\nAVERAGE PRIORITY BY RISK & CLUSTER:")
print("-" * 60)
print(f"{'Risk Level':<15} {'Cluster':<12} {'Avg Priority':<15} {'Students':<12}")
print("-" * 60)
for row in cursor.fetchall():
    risk, cluster, avg_priority, count = row
    print(f"{risk:<15} Cluster {cluster:<5} {avg_priority:>6.2f}/15{'':<7} {count:<12}")

conn.close()
