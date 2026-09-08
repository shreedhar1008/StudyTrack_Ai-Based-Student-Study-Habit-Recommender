"""
StudyTrack AI - Database Adapter Shim (Backward Compatibility)
Forwards all database functions to backend.database.adapter
"""

from backend.database.adapter import (
    get_db,
    init_db,
    is_postgres,
    get_pg_pool,
    get_database_url,
    PostgresConnectionWrapper,
    PostgresCursorWrapper
)

__all__ = [
    'get_db',
    'init_db',
    'is_postgres',
    'get_pg_pool',
    'get_database_url',
    'PostgresConnectionWrapper',
    'PostgresCursorWrapper'
]
