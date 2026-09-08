"""
StudyTrack AI - SQLite to PostgreSQL Data Migration Tool
Transfers all tables and rows from local SQLite (studytrack.db) to Cloud PostgreSQL (Neon/Supabase).
"""

import os
import sys
import sqlite3
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

load_dotenv()

try:
    from backend.database import adapter as db_adapter
except ImportError:
    import db_adapter

def migrate():
    pg_url = db_adapter.get_database_url()
    if not pg_url.startswith('postgresql://'):
        print("❌ Error: DATABASE_URL in .env is not set to a PostgreSQL URL.")
        print("Please add DATABASE_URL=postgresql://... to your .env file first.")
        return

    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sqlite_path = os.path.join(root_dir, 'database', 'studytrack.db')
    if not os.path.exists(sqlite_path):
        sqlite_path = os.path.join(root_dir, 'studytrack.db')
    if not os.path.exists(sqlite_path):
        print(f"❌ Error: {sqlite_path} does not exist.")
        return

    print("🚀 Initializing PostgreSQL tables...")
    db_adapter.init_db()

    print(f"📦 Connecting to local {sqlite_path}...")
    s_conn = sqlite3.connect(sqlite_path)
    s_conn.row_factory = sqlite3.Row
    s_cur = s_conn.cursor()

    p_conn = db_adapter.get_db()
    p_cur = p_conn.cursor()

    tables = [
        ('users', 'id'),
        ('students', 'student_id'),
        ('daily_habits', 'id'),
        ('predictions', 'id'),
        ('recommendations', 'id'),
        ('feedback', 'id'),
        ('student_habits', 'id'),
        ('admin_feedback', 'id'),
        ('user_sessions', 'id'),
        ('model_retraining_history', 'id')
    ]

    for table, pk in tables:
        try:
            s_cur.execute(f"SELECT * FROM {table}")
            rows = s_cur.fetchall()
            if not rows:
                print(f"  - {table}: 0 rows (skipped)")
                continue

            columns = [col[0] for col in s_cur.description]
            cols_str = ", ".join(columns)
            placeholders = ", ".join(["%s"] * len(columns))

            print(f"  - Migrating {len(rows)} rows into {table}...")
            raw_cur = p_cur.cursor
            insert_query = f"INSERT INTO {table} ({cols_str}) VALUES %s ON CONFLICT DO NOTHING"
            
            # Prepare data tuples
            data_tuples = [[r[c] for c in columns] for r in rows]
            
            # Fast bulk insert
            psycopg2.extras.execute_values(raw_cur, insert_query, data_tuples, page_size=1000)
            p_conn.commit()
            print(f"    ✅ {table} migrated successfully!")

            # Update PostgreSQL sequence to max ID
            try:
                p_cur.execute(f"SELECT setval(pg_get_serial_sequence('{table}', '{pk}'), COALESCE(MAX({pk}), 1)) FROM {table}")
                p_conn.commit()
            except Exception:
                pass

        except Exception as e:
            print(f"    ⚠️ Warning on {table}: {e}")

    s_conn.close()
    p_conn.close()
    print("\n🎉 Migration from SQLite to PostgreSQL completed successfully!")

if __name__ == '__main__':
    migrate()
