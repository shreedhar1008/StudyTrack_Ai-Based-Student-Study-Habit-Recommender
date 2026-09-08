"""
StudyTrack AI - Unified Database Adapter
Supports both SQLite (Local Development) and PostgreSQL (Production / Neon.tech / Supabase).
"""

import os
import sqlite3
import threading
from datetime import datetime, date
from werkzeug.security import generate_password_hash

try:
    import psycopg2
    import psycopg2.extras
    from psycopg2 import pool
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False

_pg_pool = None
_pool_lock = threading.Lock()

def get_pg_pool(db_url):
    global _pg_pool
    if _pg_pool is None:
        with _pool_lock:
            if _pg_pool is None:
                clean_url = db_url.replace('&channel_binding=require', '').replace('?channel_binding=require', '')
                if '?' not in clean_url and '&' in clean_url:
                    clean_url = clean_url.replace('&', '?', 1)
                _pg_pool = pool.ThreadedConnectionPool(
                    minconn=2,
                    maxconn=20,
                    dsn=clean_url,
                    connect_timeout=10,
                    keepalives=1,
                    keepalives_idle=30,
                    keepalives_interval=10,
                    keepalives_count=5
                )
    return _pg_pool


def get_database_url():
    """Retrieve and normalize DATABASE_URL from environment"""
    url = os.getenv('DATABASE_URL', '').strip()
    if url.startswith('postgres://'):
        url = 'postgresql://' + url[len('postgres://'):]
    return url


def is_postgres():
    """Check if the active database is PostgreSQL"""
    url = get_database_url()
    return bool(url and url.startswith('postgresql://'))


def adapt_date_iso(val):
    return val.isoformat()

def adapt_datetime_iso(val):
    return val.isoformat()

sqlite3.register_adapter(date, adapt_date_iso)
sqlite3.register_adapter(datetime, adapt_datetime_iso)


class PostgresCursorWrapper:
    """Wraps psycopg2 cursor to provide SQLite-compatible behavior (? placeholders, lastrowid, Dict/Index access)"""
    def __init__(self, cursor, conn):
        self.cursor = cursor
        self.conn = conn
        self._lastrowid = None

    def execute(self, query, params=None):
        pg_query = query.replace('?', '%s')
        is_insert = pg_query.strip().upper().startswith('INSERT INTO')
        has_returning = 'RETURNING' in pg_query.upper()

        if is_insert and not has_returning:
            pg_query += ' RETURNING *'

            if params:
                self.cursor.execute(pg_query, params)
            else:
                self.cursor.execute(pg_query)

            try:
                row = self.cursor.fetchone()
                if row:
                    self._lastrowid = row[0]
            except Exception:
                self._lastrowid = None
            return self

        if params:
            self.cursor.execute(pg_query, params)
        else:
            self.cursor.execute(pg_query)
        return self

    def executemany(self, query, params_list):
        pg_query = query.replace('?', '%s')
        return self.cursor.executemany(pg_query, params_list)

    def fetchone(self):
        return self.cursor.fetchone()

    def fetchall(self):
        return self.cursor.fetchall()

    def fetchmany(self, size=None):
        return self.cursor.fetchmany(size) if size else self.cursor.fetchmany()

    @property
    def lastrowid(self):
        return self._lastrowid

    @property
    def rowcount(self):
        return self.cursor.rowcount

    def close(self):
        return self.cursor.close()

    def __iter__(self):
        return iter(self.cursor)


class PostgresConnectionWrapper:
    """Wraps psycopg2 connection to provide SQLite-compatible interface and connection pooling"""
    def __init__(self, conn, pool_instance=None):
        self.conn = conn
        self.pool_instance = pool_instance
        self._closed = False

    def cursor(self):
        cur = self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        return PostgresCursorWrapper(cur, self.conn)

    def commit(self):
        if not self._closed:
            return self.conn.commit()

    def rollback(self):
        if not self._closed:
            return self.conn.rollback()

    def close(self):
        if not self._closed:
            self._closed = True
            if self.pool_instance is not None:
                try:
                    self.conn.rollback()
                    self.pool_instance.putconn(self.conn)
                except Exception:
                    try:
                        self.conn.close()
                    except Exception:
                        pass
            else:
                try:
                    self.conn.close()
                except Exception:
                    pass

    def execute(self, query, params=None):
        cur = self.cursor()
        return cur.execute(query, params)


def get_db():
    """Get active database connection (PostgreSQL if DATABASE_URL is set, else SQLite)"""
    db_url = get_database_url()

    if db_url.startswith('postgresql://'):
        if not PSYCOPG2_AVAILABLE:
            raise ImportError("psycopg2 is required for PostgreSQL. Run: pip install psycopg2-binary")
        p = get_pg_pool(db_url)
        try:
            conn = p.getconn()
            if conn.closed:
                clean_url = db_url.replace('&channel_binding=require', '').replace('?channel_binding=require', '')
                conn = psycopg2.connect(
                    clean_url,
                    connect_timeout=10,
                    keepalives=1,
                    keepalives_idle=30,
                    keepalives_interval=10,
                    keepalives_count=5
                )
            return PostgresConnectionWrapper(conn, p)
        except Exception:
            clean_url = db_url.replace('&channel_binding=require', '').replace('?channel_binding=require', '')
            conn = psycopg2.connect(
                clean_url,
                connect_timeout=10,
                keepalives=1,
                keepalives_idle=30,
                keepalives_interval=10,
                keepalives_count=5
            )
            return PostgresConnectionWrapper(conn, None)
    else:
        default_db = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'database', 'studytrack.db')
        if not os.path.exists(default_db) and os.path.exists('studytrack.db'):
            default_db = 'studytrack.db'
        db_path = os.getenv('SQLITE_DB_PATH', default_db)
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        conn = sqlite3.connect(db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute('PRAGMA busy_timeout = 30000')
        return conn



def init_db():
    """Initialize database with full schema and default accounts across SQLite and PostgreSQL"""
    conn = get_db()
    cursor = conn.cursor()
    use_pg = is_postgres()

    print(f"🔨 Initializing database schema on {'PostgreSQL (Cloud)' if use_pg else 'SQLite (Local)'}...")

    if not use_pg:
        conn.execute('PRAGMA journal_mode=WAL')
        conn.execute('PRAGMA synchronous=NORMAL')

    pk_type = "SERIAL PRIMARY KEY" if use_pg else "INTEGER PRIMARY KEY AUTOINCREMENT"
    ts_default = "CURRENT_TIMESTAMP"

    # 1. Users Table
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS users (
            id {pk_type},
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role TEXT DEFAULT 'student',
            created_at TIMESTAMP DEFAULT {ts_default}
        )
    """)

    # 2. Students Table
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS students (
            student_id {pk_type},
            user_id INTEGER,
            full_name TEXT NOT NULL,
            age INTEGER,
            gender TEXT,
            major TEXT,
            year INTEGER,
            gpa REAL,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    """)

    # 3. Daily Habits Table
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS daily_habits (
            id {pk_type},
            student_id INTEGER,
            log_date DATE NOT NULL,
            study_hours REAL,
            sleep_hours REAL,
            attendance_percentage REAL,
            social_media_hours REAL,
            exercise_frequency INTEGER,
            stress_level INTEGER,
            motivation_level INTEGER,
            mental_health_rating INTEGER,
            notes TEXT,
            created_at TIMESTAMP DEFAULT {ts_default},
            FOREIGN KEY (student_id) REFERENCES students (student_id),
            UNIQUE(student_id, log_date)
        )
    """)

    # 4. Predictions Table
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS predictions (
            id {pk_type},
            student_id INTEGER,
            daily_habit_id INTEGER,
            dropout_probability REAL,
            risk_level TEXT,
            cluster_number INTEGER,
            priority_score INTEGER,
            created_at TIMESTAMP DEFAULT {ts_default},
            FOREIGN KEY (student_id) REFERENCES students (student_id),
            FOREIGN KEY (daily_habit_id) REFERENCES daily_habits (id)
        )
    """)

    # 5. Recommendations Table
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS recommendations (
            id {pk_type},
            student_id INTEGER,
            prediction_id INTEGER,
            daily_habit_id INTEGER,
            category TEXT,
            message TEXT,
            priority TEXT,
            status TEXT DEFAULT 'pending',
            created_at TIMESTAMP DEFAULT {ts_default},
            FOREIGN KEY (student_id) REFERENCES students (student_id),
            FOREIGN KEY (prediction_id) REFERENCES predictions (id),
            FOREIGN KEY (daily_habit_id) REFERENCES daily_habits (id)
        )
    """)

    # 6. Feedback Table
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS feedback (
            id {pk_type},
            student_id INTEGER,
            prediction_id INTEGER,
            rating INTEGER,
            feedback_text TEXT,
            improvement_seen TEXT,
            would_recommend INTEGER,
            created_at TIMESTAMP DEFAULT {ts_default},
            FOREIGN KEY (student_id) REFERENCES students (student_id),
            FOREIGN KEY (prediction_id) REFERENCES predictions (id)
        )
    """)

    # 7. Student Habits Legacy Table
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS student_habits (
            id {pk_type},
            student_id INTEGER,
            study_hours REAL,
            sleep_hours REAL,
            attendance_percentage REAL,
            social_media_hours REAL,
            exercise_frequency INTEGER,
            stress_level INTEGER,
            motivation_level INTEGER,
            mental_health_rating INTEGER,
            recorded_at TIMESTAMP DEFAULT {ts_default},
            FOREIGN KEY (student_id) REFERENCES students (student_id)
        )
    """)

    # 8. Admin Feedback Table
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS admin_feedback (
            id {pk_type},
            admin_id INTEGER,
            student_id INTEGER,
            prediction_id INTEGER,
            effectiveness_rating INTEGER,
            prediction_accurate INTEGER,
            actual_performance TEXT,
            intervention_taken TEXT,
            feedback_text TEXT,
            created_at TIMESTAMP DEFAULT {ts_default},
            FOREIGN KEY (admin_id) REFERENCES users (id),
            FOREIGN KEY (student_id) REFERENCES students (student_id),
            FOREIGN KEY (prediction_id) REFERENCES predictions (id)
        )
    """)

    # 9. User Sessions Table
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS user_sessions (
            id {pk_type},
            user_id INTEGER,
            student_id INTEGER,
            login_time TIMESTAMP DEFAULT {ts_default},
            last_habit_log_date DATE,
            total_logins INTEGER DEFAULT 1,
            FOREIGN KEY (user_id) REFERENCES users (id),
            FOREIGN KEY (student_id) REFERENCES students (student_id)
        )
    """)

    # 10. Model Retraining History Table
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS model_retraining_history (
            id {pk_type},
            dataset_filename TEXT,
            dataset_records INTEGER,
            model_accuracy REAL,
            previous_accuracy REAL,
            improvement REAL,
            training_duration REAL,
            notes TEXT,
            training_date TIMESTAMP DEFAULT {ts_default}
        )
    """)

    cursor.execute('SELECT COUNT(*) FROM users')
    count = cursor.fetchone()[0]
    if count == 0:
        admin_password = generate_password_hash('admin123')
        student_password = generate_password_hash('student123')

        cursor.execute("""
            INSERT INTO users (id, username, email, password, role)
            VALUES (1, 'admin', 'admin@studytrack.com', ?, 'admin')
        """, (admin_password,))

        cursor.execute("""
            INSERT INTO users (id, username, email, password, role)
            VALUES (2, 'student1', 'student1@example.com', ?, 'student')
        """, (student_password,))

        cursor.execute("""
            INSERT INTO students (student_id, user_id, full_name, age, gender, major, year, gpa)
            VALUES (1, 2, 'John Doe', 20, 'Male', 'Computer Science', 2, 3.2)
        """)
        print("✅ Default users created successfully")

    # Performance Indexes
    indexes = [
        "CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)",
        "CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)",
        "CREATE INDEX IF NOT EXISTS idx_students_user_id ON students(user_id)",
        "CREATE INDEX IF NOT EXISTS idx_daily_habits_student_id ON daily_habits(student_id)",
        "CREATE INDEX IF NOT EXISTS idx_predictions_student_id ON predictions(student_id)",
        "CREATE INDEX IF NOT EXISTS idx_recommendations_daily_habit_id ON recommendations(daily_habit_id)"
    ]
    for idx_sql in indexes:
        try:
            cursor.execute(idx_sql)
        except Exception:
            pass

    conn.commit()
    conn.close()
    print("✅ Database initialization complete!")
